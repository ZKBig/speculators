#!/usr/bin/env python3
# ruff: noqa: T201, S603, BLE001, C901, PLR2004, PLW1510
# A cluster ops launcher, not library code: it prints progress and shells out.
"""Regenerate an on-policy corpus with a Gemma 4 target on one 8-GPU pod.

Serves the target with vLLM (data parallel, one replica per GPU: 26B-A4B in
bf16 is ~52 GB, so a replica fits an 80 GB card) and runs
scripts/response_regeneration/script.py over every preset in G4_PRESETS,
one JSONL per preset. Resumable: script.py --resume skips rows already in
the outfile, so a restarted job continues. Only LOCAL_RANK 0 works; the
other ranks exit at once (the scheduler hands out 8 processes per pod).

Follow with a prep job (k8s_dflash2_launch.py, DF2_PREP_ONLY=1) pointed at
G4_OUT_DIR to tokenize for the same target; training then runs online.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


MODEL = env("G4_MODEL", "google/gemma-4-26b-a4b-it")
OUT_DIR = Path(env("G4_OUT_DIR", "/gpfs/zwang33/gemma4/regen"))
PRESETS = env("G4_PRESETS", "open_perfectblend").split()
DP_SIZE = env("G4_DP_SIZE", "8")
MAX_MODEL_LEN = env("G4_MAX_MODEL_LEN", "16384")
MAX_TOKENS = env("G4_MAX_TOKENS", "8192")
CONCURRENCY = env("G4_CONCURRENCY", "256")
LIMIT = env("G4_LIMIT", "")  # rows per preset; unset = whole preset
SAMPLING = env("G4_SAMPLING_PARAMS", "")  # JSON forwarded to --sampling-params
PORT = int(env("G4_VLLM_PORT", "8300"))
VLLM_PY = env("G4_VLLM_PY", "/gpfs/zwang33/venv_vllm/bin/python")
REPO = Path(env("G4_REPO", "/gpfs/zwang33/speculators"))
HEALTH_TIMEOUT_S = int(env("G4_HEALTH_TIMEOUT_S", "3600"))


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    if int(env("LOCAL_RANK", "0")) != 0:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    serve = [
        VLLM_PY,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL,
        "--port",
        str(PORT),
        "--data-parallel-size",
        DP_SIZE,
        "--max-model-len",
        MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        env("G4_GPU_MEM_UTIL", "0.90"),
        # Text-only: the HF repo is multimodal; no image slots keeps the
        # vision tower out of memory and the chat template text-only.
        "--limit-mm-per-prompt",
        '{"image": 0, "audio": 0}',
    ]
    extra = env("G4_VLLM_EXTRA_ARGS", "")
    if extra:
        serve += extra.split()
    print("[regen] $", " ".join(serve), flush=True)
    server = subprocess.Popen(serve, cwd=REPO)
    try:
        t0 = time.time()
        while not healthy():
            if server.poll() is not None:
                print("[regen] vLLM exited before becoming healthy", flush=True)
                return server.returncode or 1
            if time.time() - t0 > HEALTH_TIMEOUT_S:
                print("[regen] vLLM health timeout", flush=True)
                return 1
            time.sleep(10)
        print(f"[regen] vLLM healthy after {time.time() - t0:.0f}s", flush=True)

        tag = MODEL.split("/")[-1]
        for preset in PRESETS:
            outfile = OUT_DIR / f"{preset}_{tag}.jsonl"
            cmd = [
                sys.executable,
                "scripts/response_regeneration/script.py",
                "--endpoint",
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                "--model",
                MODEL,
                "--dataset",
                preset,
                "--outfile",
                str(outfile),
                "--concurrency",
                CONCURRENCY,
                "--max-tokens",
                MAX_TOKENS,
                "--resume",
            ]
            if LIMIT:
                cmd += ["--limit", LIMIT]
            if SAMPLING:
                cmd += ["--sampling-params", SAMPLING]
            print(f"[regen] {preset} -> {outfile}", flush=True)
            print("[regen] $", " ".join(cmd), flush=True)
            rc = subprocess.run(cmd, cwd=REPO).returncode
            if rc != 0:
                print(f"[regen] {preset} failed rc={rc}", flush=True)
                return rc
        print("[regen] all presets done", flush=True)
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
