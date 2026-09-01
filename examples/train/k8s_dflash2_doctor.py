#!/usr/bin/env python3
# ruff: noqa: T201, S104, S310, S603, S607, S108, BLE001, PLR2004, PLW0603, PLW1510, PLC0415, PTH112
"""Preflight checks for the DFlash2 k8s job.

Every check here corresponds to a way the real job has stalled or died, and each
one is cheap. Run it in the pod before submitting a multi-hour job:

    python examples/train/k8s_dflash2_doctor.py

It can also be dropped in as the job's `mainProgram` for a one-shot diagnostic
run: under torchrun only rank 0 reports, so the output stays readable.
"""

import os
import shutil
import socket
import subprocess
import urllib.request
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RANK = int(os.environ.get("RANK") or 0)
LOCAL_RANK = int(os.environ.get("LOCAL_RANK") or 0)

MODEL = os.environ.get("DF2_MODEL") or "Qwen/Qwen3-4B"
DATA_DIR = Path(os.environ.get("DF2_DATA_DIR") or "/gpfs/zwang33/dflash2/data")
VLLM_PY = Path(
    os.environ.get("DF2_VLLM_PY")
    or (os.environ.get("DF2_VLLM_VENV") or "/gpfs/zwang33/venv_vllm") + "/bin/python"
)
HF_HOME = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))

_fail = 0


def report(ok: bool, name: str, detail: str = "") -> None:
    global _fail
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def check_loopback() -> None:
    """Bind and connect on 127.0.0.1.

    This is exactly what torch's TCPStore does when world_size=1: rank 0 listens
    and then connects to itself. When the container's loopback is unusable the
    only symptom in the real job is a silent ten-minute stall followed by
    "client socket has timed out".
    """
    try:
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        cli = socket.socket()
        cli.settimeout(5)
        cli.connect(("127.0.0.1", port))
        cli.close()
        srv.close()
        report(True, "loopback bind+connect", f"127.0.0.1:{port}")
    except Exception as e:
        report(False, "loopback bind+connect", repr(e))


def check_lo_interface() -> None:
    try:
        out = subprocess.run(
            ["ip", "addr", "show", "lo"], capture_output=True, text=True
        ).stdout
        up = "state UP" in out or "LOOPBACK,UP" in out
        has_ip = "127.0.0.1/8" in out
        report(
            up and has_ip,
            "lo interface",
            out.strip().splitlines()[0] if out else "no output",
        )
    except FileNotFoundError:
        report(True, "lo interface", "skipped, no `ip` in image")


def check_tcpstore() -> None:
    """The real thing: torch's own store, the call that timed out in the job."""
    try:
        import torch.distributed as dist
    except Exception as e:
        report(False, "torch TCPStore", f"torch not importable: {e!r}")
        return
    try:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        store = dist.TCPStore(
            "127.0.0.1",
            port,
            1,
            is_master=True,
            timeout=timedelta(seconds=15),
        )
        store.set("k", "v")
        ok = store.get("k") == b"v"
        report(ok, "torch TCPStore on loopback", f"port {port}")
    except Exception as e:
        report(False, "torch TCPStore on loopback", repr(e))


def check_pod_ip() -> None:
    """The fallback address, if loopback is the problem: DF2_VLLM_HOST."""
    try:
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        cli = socket.socket()
        cli.settimeout(5)
        cli.connect((ip, port))
        cli.close()
        srv.close()
        report(True, "pod IP bind+connect", f"{host} -> {ip}:{port}")
    except Exception as e:
        report(False, "pod IP bind+connect", repr(e))


def check_gpus() -> None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        busy = [
            ln
            for ln in out.splitlines()
            if int(ln.split(",")[1].strip().split()[0]) > 500
        ]
        report(
            not busy,
            "GPUs idle",
            f"{len(out.splitlines())} GPUs; in use: {busy or 'none'}",
        )
    except Exception as e:
        report(False, "GPUs idle", repr(e))


def check_stale_vllm() -> None:
    """A previous failed run can leave a verifier holding GPU 0 and its ports."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,cmd"], capture_output=True, text=True
        ).stdout
        hits = [
            ln for ln in out.splitlines() if "vllm" in ln.lower() and "grep" not in ln
        ]
        report(not hits, "no stale vLLM process", f"{len(hits)} found" if hits else "")
        for h in hits[:5]:
            print("        ", h.strip()[:120])
    except Exception as e:
        report(False, "no stale vLLM process", repr(e))


def check_verifier_venv() -> None:
    if not VLLM_PY.exists():
        report(False, "verifier interpreter", f"{VLLM_PY} missing")
        return
    out = subprocess.run(
        [
            str(VLLM_PY),
            "-c",
            "import vllm, torch; print(vllm.__version__, torch.__version__)",
        ],
        capture_output=True,
        text=True,
    )
    report(
        out.returncode == 0,
        "verifier venv imports vllm",
        (out.stdout or out.stderr).strip().splitlines()[-1]
        if (out.stdout or out.stderr)
        else "",
    )


def check_model_cache() -> None:
    tag = "models--" + MODEL.replace("/", "--")
    hit = list(HF_HOME.glob(f"**/{tag}"))
    if Path(MODEL).exists():
        report(True, "model available", f"local path {MODEL}")
        return
    report(
        bool(hit),
        "model in HF cache",
        str(hit[0]) if hit else f"{tag} not under {HF_HOME}",
    )
    incomplete = list(HF_HOME.glob("**/*.incomplete"))
    if incomplete:
        print(
            f"         ({len(incomplete)} .incomplete files -- a download is in flight)"
        )


def check_hf_network() -> None:
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/{MODEL}/resolve/main/config.json", method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            report(r.status < 400, "huggingface.co reachable", str(r.status))
    except Exception as e:
        report(False, "huggingface.co reachable", repr(e))


def check_data() -> None:
    prepared = (DATA_DIR / "dataset_info.json").exists()
    src = os.environ.get("DF2_TRAIN_SRC") or ""
    report(
        prepared or bool(src),
        "data available",
        f"prepared at {DATA_DIR}"
        if prepared
        else f"will build from {src or '<unset>'}",
    )


def check_disk() -> None:
    for p in ("/gpfs", "/tmp"):
        if os.path.isdir(p):
            usage = shutil.disk_usage(p)
            free_gb = usage.free / 1e9
            report(free_gb > 50, f"free space on {p}", f"{free_gb:.0f} GB")


def main() -> int:
    if RANK != 0:
        return 0
    print(f"=== DFlash2 preflight (host {socket.gethostname()}, rank {RANK}) ===")
    print(f"    repo         {REPO}")
    print(f"    model        {MODEL}")
    print(f"    verifier py  {VLLM_PY}")
    print(f"    HF_HOME      {HF_HOME}")
    print(f"    data dir     {DATA_DIR}")
    print()
    check_loopback()
    check_lo_interface()
    check_tcpstore()
    check_pod_ip()
    check_stale_vllm()
    check_gpus()
    check_verifier_venv()
    check_model_cache()
    check_hf_network()
    check_data()
    check_disk()
    print()
    if _fail:
        print(f"{_fail} check(s) failed -- fix these before submitting the real job.")
    else:
        print("all checks passed.")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
