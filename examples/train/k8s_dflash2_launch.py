#!/usr/bin/env python3
# ruff: noqa: T201, S108, S603, BLE001, C901, PLR2004, PLW1510, PTH103, PTH118
# A cluster ops launcher, not library code: it prints progress, shells out, and
# writes to /tmp on purpose. Linting it as library code fights every one of those.
"""Launch DFlash2 online training on the k8s job framework.

The framework runs `torchrun --nnodes=N --nproc_per_node=<gpus>` over this file, so
every process here owns ONE GPU. LOCAL_RANK 0 serves the verifier (vLLM) for its
node; the rest are trainer ranks. That is a better split than the standalone bash
recipe's 4 vLLM + 4 training: one server saturates easily, so giving it a single GPU
leaves seven for the trainers instead of four.

Online training: hidden states are pulled from the live server and deleted after use
(--on-missing generate --on-generate delete), so only the PREPARED token data has to
exist beforehand.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


RANK = int(env("RANK", "0"))
WORLD = int(env("WORLD_SIZE", "1"))
LOCAL_RANK = int(env("LOCAL_RANK", "0"))
LOCAL_WORLD = int(env("LOCAL_WORLD_SIZE", "1"))
NODE = int(env("GROUP_RANK", "0"))
NUM_NODES = max(1, WORLD // max(1, LOCAL_WORLD))
MASTER_ADDR = env("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = int(env("MASTER_PORT", "23456"))

IS_VERIFIER = LOCAL_RANK == 0
TRAINERS_PER_NODE = max(1, LOCAL_WORLD - 1)
TRAIN_WORLD = NUM_NODES * TRAINERS_PER_NODE
TRAIN_RANK = NODE * TRAINERS_PER_NODE + (LOCAL_RANK - 1)
TRAIN_PORT = int(env("DF2_TRAIN_PORT", str(MASTER_PORT + 1)))

MODEL = env("DF2_MODEL", "Qwen/Qwen3-4B")
OUT_ROOT = Path(env("DF2_OUT_ROOT", "/gpfs/zwang33/dflash2"))
OUTPUT_DIR = OUT_ROOT / env("DF2_RUN_DIR", "dflash2_qwen3_4b_8spec")
DATA_DIR = Path(env("DF2_DATA_DIR", str(OUTPUT_DIR / "data")))
VAL_DATA_DIR = env("DF2_VAL_DATA_DIR", "")
# Raw corpus to tokenize when DATA_DIR is not already prepared: a local
# JSON/JSONL path or an "hf:org/dataset" spec, exactly as prepare_data.py takes.
TRAIN_SRC = env("DF2_TRAIN_SRC", "")
MAX_SAMPLES = env("DF2_MAX_SAMPLES", "")
PREP_WORKERS = env("DF2_PREP_WORKERS", "32")

SEQ_LENGTH = env("DF2_SEQ_LENGTH", "8192")
TARGET_LAYER_IDS = env("DF2_TARGET_LAYER_IDS", "1 9 17 25 33").split()
BLOCK_SIZE = env("DF2_BLOCK_SIZE", "9")  # 9 - 1 = 8 speculative tokens
MAX_ANCHORS = env("DF2_MAX_ANCHORS", "512")
NUM_LAYERS = env("DF2_NUM_LAYERS", "5")
CONV_KERNEL_SIZE = env("DF2_CONV_KERNEL_SIZE", "2")
CONV_GROUP_SIZE = env("DF2_CONV_GROUP_SIZE", "16")
SELECTOR_RANK = env("DF2_SELECTOR_RANK", "256")
SELECTOR_TOP_K = env("DF2_SELECTOR_TOP_K", "16")
LOSS_FN = env("DF2_LOSS_FN", '{"ce": 0.1, "tv": 0.9}')
DECAY_GAMMA = env("DF2_DECAY_GAMMA", "4.0")
EPOCHS = env("DF2_EPOCHS", "1")
LR = env("DF2_LR", "6e-4")
WEIGHT_DECAY = env("DF2_WEIGHT_DECAY", "0.0")
WARMUP_RATIO = env("DF2_WARMUP_RATIO", "0.04")
CHECKPOINT_FREQ = env("DF2_CHECKPOINT_FREQ", "0.1")
LOG_FREQ = env("DF2_LOG_FREQ", "50")
RUN_NAME = env("DF2_RUN_NAME", "dflash2-qwen3-4b-8spec")

VLLM_PORT = int(env("DF2_VLLM_PORT", "8300"))
# Host the trainers and the health check dial. Loopback by default; set it to
# the pod's own address if the container's "lo" is not usable, which shows up
# as every connection to 127.0.0.1 timing out.
VLLM_HOST = env("DF2_VLLM_HOST", "127.0.0.1")
VLLM_VENV = Path(env("DF2_VLLM_VENV", "/gpfs/zwang33/venv_vllm"))
VLLM_PY = Path(env("DF2_VLLM_PY", str(VLLM_VENV / "bin" / "python")))

DONE = OUTPUT_DIR / ".train_done"
READY = OUTPUT_DIR / ".prep_ready"
STAMP = DATA_DIR / ".data_ready"

_DIST_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    # torchrun sets TORCHELASTIC_USE_AGENT_STORE=True for its workers, and torch's
    # TCP rendezvous reads it: with it set, init_process_group builds a CLIENT
    # store and waits for the agent to be listening, even at world_size=1 where
    # rank 0 would otherwise create the server itself. A child that is not part of
    # the job -- vLLM here -- then waits ten minutes for a listener that does not
    # exist. Constructing a TCPStore directly bypasses the rendezvous, which is
    # why a store probe passes in the same interpreter that vLLM stalls in.
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_ERROR_FILE",
    "TORCH_ELASTIC_WORKER_IDENTIFIER",
)


def strip_dist(e: dict) -> dict:
    """Drop the framework's rendezvous vars from a child's environment."""
    for v in _DIST_VARS:
        e.pop(v, None)
    return e


def child_env(**extra: str) -> dict:
    """Environment for every subprocess."""
    e = dict(os.environ)
    roots = [str(REPO / "src"), str(REPO / "hs_connectors" / "src")]
    e["PYTHONPATH"] = os.pathsep.join(
        roots + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else [])
    )

    # Give each NODE its own compile cache. The job spec points these at shared
    # /gpfs, where inductor's write-then-rename is not atomic across nodes: a rank
    # can dlopen a .so another node is still writing and die with "file too short".
    for _var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        _root = e.get(_var)
        if _root:
            _d = os.path.join(_root, socket.gethostname())
            os.makedirs(_d, exist_ok=True)
            e[_var] = _d

    # Inductor holds the GIL for the whole compile, so NCCL's watchdog thread cannot
    # tick; the heartbeat monitor reads that as a wedged rank and aborts the job.
    # Set it HERE rather than trusting the pod spec, which has failed to reach the
    # process before.
    e["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = env("DF2_NCCL_HEARTBEAT_SEC", "3600")
    e.update(extra)
    return e


def vllm_healthy() -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=5
        ) as r:
            return r.status == 200
    except Exception:
        return False


def data_is_prepared() -> bool:
    return (DATA_DIR / "dataset_info.json").exists()


def check_inputs() -> None:
    """Fail loudly, before any GPU work, on the things that silently misbehave."""
    if not data_is_prepared() and not TRAIN_SRC:
        sys.exit(
            f"[r{RANK}] FATAL - {DATA_DIR} is not prepare_data.py output (no "
            "dataset_info.json) and DF2_TRAIN_SRC is unset, so there is nothing to "
            "build it from. Online training generates only the HIDDEN STATES on the "
            "fly; the tokenized data still has to exist. Set DF2_TRAIN_SRC to a "
            'JSONL path or an "hf:org/dataset" spec, or point DF2_DATA_DIR at an '
            "already-prepared corpus."
        )
    if not VLLM_PY.exists():
        sys.exit(
            f"[r{RANK}] FATAL - verifier interpreter {VLLM_PY} not found; "
            "set DF2_VLLM_VENV to a venv that has vllm installed."
        )


def serve_vllm_until_done() -> int:
    """Verifier role (LOCAL_RANK 0): serve on GPU 0 until the trainers are done.

    The framework's torchrun waits for every worker, so this process must exit on
    its own once training finishes -- train rank 0 writes DONE for that.
    """
    # The hidden-state spool is the verifier->trainer channel. Leftovers from a
    # crashed run would be read back as if freshly generated (--on-missing generate),
    # silently training on stale activations.
    spool = Path(env("DF2_HIDDEN_STATES_DIR", "/tmp/hidden_states_dflash2"))
    if spool.exists():
        print(f"[r{RANK}] clearing stale spool {spool}", flush=True)
        shutil.rmtree(spool, ignore_errors=True)

    # vLLM runs its OWN single-GPU process group. Inheriting the job's rendezvous
    # makes it try to join the job-wide group instead, and it then waits for ranks
    # that are themselves waiting for it.
    e = strip_dist(
        child_env(
            CUDA_VISIBLE_DEVICES="0",
            # One process on one GPU, so its internal store belongs on loopback. Left to
            # guess, vLLM picks the pod's routable IP and its own EngineCore cannot dial
            # back to it (10-minute TCPStore timeout).
            VLLM_HOST_IP=env("DF2_VLLM_HOST_IP", "127.0.0.1"),
            HOST_IP=env("DF2_VLLM_HOST_IP", "127.0.0.1"),
            # NCCL settings tuned for the trainer fabric do not apply to a single-GPU
            # engine and can send it hunting for IB devices.
            NCCL_IB_DISABLE="1",
            NCCL_P2P_DISABLE="1",
            VLLM_USE_FLASHINFER_SAMPLER="0",
            VLLM_ATTENTION_BACKEND=env("DF2_VLLM_ATTN_BACKEND", "FLASH_ATTN"),
        )
    )
    # Optional, and only applied when non-empty. Everything this launcher adds
    # over the XPress one lives here, so setting them all empty reduces the
    # verifier to that launcher's exact invocation -- which is the only way to
    # separate "our extra flags broke it" from "the environment changed".
    #   DF2_VLLM_V1_MP=0     run EngineCore in-process (no loopback TCPStore)
    #   DF2_VLLM_USE_LIBUV=0 legacy store backend; libuv's listener can fail
    #                        silently in a restricted container
    for _var, _key in (
        ("DF2_VLLM_V1_MP", "VLLM_ENABLE_V1_MULTIPROCESSING"),
        ("DF2_VLLM_USE_LIBUV", "USE_LIBUV"),
    ):
        _val = env(_var, "")
        if _val:
            e[_key] = _val
            print(f"[r{RANK}] verifier env {_key}={_val}", flush=True)

    cmd = [
        str(VLLM_PY),
        "scripts/launch_vllm.py",
        MODEL,
        "--target-layer-ids",
        *TARGET_LAYER_IDS,
        "--port",
        str(VLLM_PORT),
    ]
    bind = env("DF2_VLLM_BIND", "")
    if bind:
        cmd += ["--host", bind]
    if env("DF2_VLLM_MAX_MODEL_LEN", "1") == "1":
        cmd += ["--max-model-len", str(int(SEQ_LENGTH) + 2)]
    print(f"[r{RANK}] node {NODE}: verifier on GPU 0 ({VLLM_PY})", flush=True)
    proc = subprocess.Popen(cmd, env=e, cwd=REPO)
    for i in range(360):  # up to 30 min
        if i and i % 24 == 0:  # heartbeat every 2 min
            print(f"[r{RANK}] node {NODE}: vLLM still starting ({i * 5}s)", flush=True)
        if proc.poll() is not None:
            sys.exit(f"[r{RANK}] FATAL: vLLM exited during startup ({proc.returncode})")
        if vllm_healthy():
            break
        time.sleep(5)
    else:
        proc.kill()
        sys.exit(f"[r{RANK}] FATAL: vLLM not ready in time")
    print(f"[r{RANK}] node {NODE}: vLLM ready", flush=True)
    try:
        if RANK == 0:
            # Tokenizing needs the render endpoint, which is THIS server: upstream
            # now derives loss masks from vLLM's chat template rather than a regex.
            # So prep runs here, after health, and the trainers wait on READY.
            # Inside the try: a prep failure must still take the server down with
            # it, or the process exits and leaves vLLM holding GPU 0.
            prepare()
        print(f"[r{RANK}] node {NODE}: serving until training completes", flush=True)
        while not DONE.exists():
            if proc.poll() is not None:
                sys.exit(
                    f"[r{RANK}] FATAL: vLLM died while serving ({proc.returncode})"
                )
            time.sleep(30)
        print(f"[r{RANK}] node {NODE}: training done, stopping verifier", flush=True)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


def prepare() -> None:
    """Rank 0 only: tokenize the corpus unless it is already prepared."""
    want = f"{TRAIN_SRC}|{MAX_SAMPLES or 'all'}|{SEQ_LENGTH}"
    if data_is_prepared():
        if STAMP.exists() and STAMP.read_text() != want:
            sys.exit(
                f"[r{RANK}] FATAL - {DATA_DIR} was built with different settings.\n"
                f"  want: {want}\n  have: {STAMP.read_text()}\n"
                "Point DF2_DATA_DIR somewhere else rather than overwriting it."
            )
        print(f"[r{RANK}] [skip] {DATA_DIR} already prepared", flush=True)
        READY.write_text("ok")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STAMP.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "scripts/prepare_data.py",
        "--model",
        MODEL,
        "--data",
        TRAIN_SRC,
        "--output",
        str(DATA_DIR),
        "--seq-length",
        SEQ_LENGTH,
        # NOT the /v1-suffixed URL the trainer uses: prepare_data appends
        # /v1/chat/completions/render itself, so the /v1 form 404s.
        "--render-endpoint",
        f"http://{VLLM_HOST}:{VLLM_PORT}",
        "--num-preprocessing-workers",
        PREP_WORKERS,
        "--overwrite",
    ]
    if MAX_SAMPLES:
        cmd += ["--max-samples", MAX_SAMPLES]
    print(f"[r{RANK}] preparing {TRAIN_SRC} -> {DATA_DIR}", flush=True)
    print(f"[r{RANK}] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=child_env(), cwd=REPO).returncode
    if rc != 0:
        sys.exit(f"[r{RANK}] FATAL: prepare_data.py failed ({rc})")
    STAMP.write_text(want)
    READY.write_text("ok")


def wait_for_prep(timeout_s: int = 14400) -> None:
    print(f"[r{RANK}] waiting for rank 0 to finish data prep", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if READY.exists():
            print(f"[r{RANK}] prep ready", flush=True)
            return
        time.sleep(15)
    sys.exit(f"[r{RANK}] FATAL: rank 0 did not finish prep within {timeout_s}s")


def train() -> int:
    """Trainer role (LOCAL_RANK >= 1): one rank of the trainer group.

    LOCAL_RANK doubles as the device index in speculators, so keeping the physical
    GPU number here puts this rank on its own GPU and leaves GPU 0 to the verifier.
    """
    e = strip_dist(child_env())
    e.update(
        RANK=str(TRAIN_RANK),
        WORLD_SIZE=str(TRAIN_WORLD),
        LOCAL_RANK=str(LOCAL_RANK),
        MASTER_ADDR=MASTER_ADDR,
        MASTER_PORT=str(TRAIN_PORT),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--verifier-name-or-path",
        MODEL,
        "--speculator-type",
        "dflash2",
        "--data-path",
        str(DATA_DIR),
        "--vllm-endpoint",
        f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        "--save-path",
        str(OUTPUT_DIR / "checkpoints"),
        "--block-size",
        BLOCK_SIZE,
        "--max-anchors",
        MAX_ANCHORS,
        "--num-layers",
        NUM_LAYERS,
        "--target-layer-ids",
        *TARGET_LAYER_IDS,
        "--conv-kernel-size",
        CONV_KERNEL_SIZE,
        "--conv-group-size",
        CONV_GROUP_SIZE,
        "--selector-rank",
        SELECTOR_RANK,
        "--selector-top-k",
        SELECTOR_TOP_K,
        "--loss-fn",
        LOSS_FN,
        "--dflash-decay-gamma",
        DECAY_GAMMA,
        "--per-position-loss-weight",
        "fixed-exp-decay",
        "--optimizer",
        "adamw",
        "--lr",
        LR,
        "--weight-decay",
        WEIGHT_DECAY,
        "--scheduler-type",
        "cosine",
        "--scheduler-warmup-ratio",
        WARMUP_RATIO,
        "--epochs",
        EPOCHS,
        "--total-seq-len",
        SEQ_LENGTH,
        "--seed",
        "42",
        "--fsdp-shard",
        "--on-missing",
        "generate",
        "--on-generate",
        "delete",
        "--checkpoint-freq",
        CHECKPOINT_FREQ,
        "--log-freq",
        LOG_FREQ,
        "--logger",
        "wandb",
        "--run-name",
        RUN_NAME,
    ]
    if VAL_DATA_DIR:
        cmd += ["--val-data-path", VAL_DATA_DIR]
    if TRAIN_RANK == 0:
        print(
            f"[r{RANK}] trainer group: {TRAIN_WORLD} ranks "
            f"({NUM_NODES} nodes x {TRAINERS_PER_NODE}), rdzv "
            f"{MASTER_ADDR}:{TRAIN_PORT}",
            flush=True,
        )
        print(f"[r{RANK}] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=e, cwd=REPO).returncode
    if TRAIN_RANK == 0 and rc == 0:
        # Releases every verifier; without it their torchrun workers never exit.
        DONE.write_text("ok")
    return rc


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RANK == 0:
        # A rerun must not inherit stale markers.
        DONE.unlink(missing_ok=True)
        READY.unlink(missing_ok=True)
    check_inputs()
    print(
        f"[r{RANK}] node {NODE}/{NUM_NODES} gpu {LOCAL_RANK} "
        f"role={'verifier' if IS_VERIFIER else f'trainer#{TRAIN_RANK}'}",
        flush=True,
    )
    if IS_VERIFIER:
        return serve_vllm_until_done()
    wait_for_prep()
    return train()


if __name__ == "__main__":
    raise SystemExit(main())
