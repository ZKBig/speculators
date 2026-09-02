#!/usr/bin/env python3
# ruff: noqa: T201, S104, S110, S310, S603, S607, S108, BLE001, PLR2004, PLW0603, PLW1510, PLC0415, PTH112
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
import time
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
VLLM_PORT = int(os.environ.get("DF2_VLLM_PORT") or 8300)
VLLM_HOST = os.environ.get("DF2_VLLM_HOST") or "127.0.0.1"
TARGET_LAYER_IDS = (os.environ.get("DF2_TARGET_LAYER_IDS") or "1 9 17 25 33").split()
SEQ_LENGTH = os.environ.get("DF2_SEQ_LENGTH") or "8192"
START_TIMEOUT = int(os.environ.get("DF2_VLLM_TIMEOUT") or 900)

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


def check_tcpstore_in_verifier_venv() -> None:
    """The same store test, but in the VERIFIER's interpreter.

    The check above runs in the job's python; vLLM's EngineCore runs in the
    verifier venv with its own torch. Since torch 2.4 the store is backed by
    libuv, and in a container that blocks what libuv needs the listener never
    comes up -- the client then times out after ten minutes, which is exactly how
    the real job stalls. So probe there too, and if it fails, say whether
    USE_LIBUV=0 is the way out.
    """
    if not VLLM_PY.exists():
        report(False, "TCPStore in verifier venv", f"{VLLM_PY} missing")
        return
    probe = (
        "import datetime, socket, torch.distributed as dist\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0));"
        " p = s.getsockname()[1]; s.close()\n"
        "st = dist.TCPStore('127.0.0.1', p, 1, is_master=True,"
        " timeout=datetime.timedelta(seconds=20))\n"
        "st.set('k', 'v'); assert st.get('k') == b'v'; print('ok')\n"
    )
    base = subprocess.run([str(VLLM_PY), "-c", probe], capture_output=True, text=True)
    if base.returncode == 0:
        report(True, "TCPStore in verifier venv", "works as-is")
        return
    err = (base.stderr or base.stdout).strip().splitlines()
    alt = subprocess.run(
        [str(VLLM_PY), "-c", probe],
        capture_output=True,
        text=True,
        env=dict(os.environ, USE_LIBUV="0"),
    )
    if alt.returncode == 0:
        report(
            False,
            "TCPStore in verifier venv",
            "fails by default, WORKS with USE_LIBUV=0 -> set DF2_VLLM_USE_LIBUV=0",
        )
    else:
        report(
            False,
            "TCPStore in verifier venv",
            f"fails with and without libuv: {err[-1] if err else '?'}",
        )


def check_rendezvous_in_verifier_venv() -> None:
    """init_process_group at world_size=1, the call vLLM's EngineCore stalls in.

    Distinct from the store probe above: this one goes through torch's TCP
    rendezvous, which reads TORCHELASTIC_USE_AGENT_STORE and, when a torchrun
    agent set it, builds a CLIENT store instead of letting rank 0 create the
    server. That is a ten-minute timeout for any child launched from a worker
    that did not clear the variable, and it is invisible to a direct TCPStore.
    """
    if not VLLM_PY.exists():
        report(False, "rendezvous in verifier venv", f"{VLLM_PY} missing")
        return
    probe = (
        "import datetime, os, socket, torch.distributed as dist\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0));"
        " p = s.getsockname()[1]; s.close()\n"
        "dist.init_process_group(backend='gloo', init_method=f'tcp://127.0.0.1:{p}',"
        " world_size=1, rank=0, timeout=datetime.timedelta(seconds=20))\n"
        "dist.destroy_process_group(); print('ok')\n"
    )
    inherited = os.environ.get("TORCHELASTIC_USE_AGENT_STORE")
    out = subprocess.run(
        [str(VLLM_PY), "-c", probe], capture_output=True, text=True, timeout=120
    )
    detail = f"TORCHELASTIC_USE_AGENT_STORE={inherited or '<unset>'}"
    if out.returncode == 0:
        report(True, "rendezvous in verifier venv", detail)
        return
    tail = (out.stderr or out.stdout).strip().splitlines()
    report(
        False, "rendezvous in verifier venv", f"{detail}; {tail[-1] if tail else '?'}"
    )


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


def check_vllm_starts() -> None:
    """Actually start the verifier and wait for /health.

    The other checks pass on a machine where the real job still stalls: they
    prove the pieces work, not that vLLM comes up. This runs the verifier under
    the SAME environment k8s_dflash2_launch.py gives it, so a stall here is the
    stall, reproduced in minutes instead of a whole 8-GPU job.
    """
    env = dict(os.environ)
    # TORCHELASTIC_USE_AGENT_STORE is the one that matters: torch's TCP rendezvous
    # reads it and builds a CLIENT store, so a child launched from a torchrun
    # worker waits for a listener the agent never puts at that address.
    for var in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_ERROR_FILE",
        "TORCH_ELASTIC_WORKER_IDENTIFIER",
    ):
        env.pop(var, None)
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(REPO / "src"), str(REPO / "hs_connectors" / "src")]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        ),
        CUDA_VISIBLE_DEVICES="0",
        VLLM_HOST_IP=os.environ.get("DF2_VLLM_HOST_IP") or "127.0.0.1",
        HOST_IP=os.environ.get("DF2_VLLM_HOST_IP") or "127.0.0.1",
        NCCL_IB_DISABLE="1",
        NCCL_P2P_DISABLE="1",
        VLLM_ENABLE_V1_MULTIPROCESSING=os.environ.get("DF2_VLLM_V1_MP") or "0",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_ATTENTION_BACKEND=os.environ.get("DF2_VLLM_ATTN_BACKEND") or "FLASH_ATTN",
    )
    libuv = os.environ.get("DF2_VLLM_USE_LIBUV")
    if libuv is not None:
        env["USE_LIBUV"] = libuv
        print(f"    USE_LIBUV={libuv}")
    cmd = [
        str(VLLM_PY),
        "scripts/launch_vllm.py",
        MODEL,
        "--target-layer-ids",
        *TARGET_LAYER_IDS,
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(int(SEQ_LENGTH) + 2),
    ]
    print(f"    starting: {' '.join(cmd)}")
    print(f"    (up to {START_TIMEOUT}s; vLLM's own output follows)")
    proc = subprocess.Popen(cmd, env=env, cwd=REPO)
    deadline = time.time() + START_TIMEOUT
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                report(
                    False,
                    "vLLM starts",
                    f"exited with {proc.returncode} during startup",
                )
                return
            try:
                with urllib.request.urlopen(
                    f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=5
                ) as r:
                    if r.status == 200:
                        took = START_TIMEOUT - int(deadline - time.time())
                        report(True, "vLLM starts", f"healthy after ~{took}s")
                        return
            except Exception:
                pass
            time.sleep(5)
        report(
            False,
            "vLLM starts",
            f"no /health after {START_TIMEOUT}s -- THIS is the stall",
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


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
    check_tcpstore_in_verifier_venv()
    check_rendezvous_in_verifier_venv()
    check_pod_ip()
    check_stale_vllm()
    check_gpus()
    check_verifier_venv()
    check_model_cache()
    check_hf_network()
    check_data()
    check_disk()
    if os.environ.get("DF2_DOCTOR_START_VLLM") == "1":
        print()
        print("=== starting the verifier for real (DF2_DOCTOR_START_VLLM=1) ===")
        check_vllm_starts()
    print()
    if _fail:
        print(f"{_fail} check(s) failed -- fix these before submitting the real job.")
    else:
        print("all checks passed.")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
