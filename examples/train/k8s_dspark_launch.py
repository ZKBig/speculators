#!/usr/bin/env python3
# ruff: noqa: T201, S108, S603, BLE001, C901, PLR2004, PLW1510, PTH103, PTH118
# A cluster ops launcher, not library code: it prints progress, shells out, and
# writes to /tmp on purpose. Linting it as library code fights every one of those.
"""Launch DSpark online training on the k8s job framework.

A copy of k8s_dflash2_launch.py with DSpark's flags in place of DFlash2's. It is
a copy rather than a shared launcher on purpose: the DFlash2 file is what a
running job re-clones on restart, and a shared one would put that job's next
restart at the mercy of an edit made for this one. Merge them once both runs are
done.

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
TRAIN_PORT = int(env("DSP_TRAIN_PORT", str(MASTER_PORT + 1)))

MODEL = env("DSP_MODEL", "Qwen/Qwen3-4B")
OUT_ROOT = Path(env("DSP_OUT_ROOT", "/gpfs/zwang33/dspark"))
OUTPUT_DIR = OUT_ROOT / env("DSP_RUN_DIR", "dspark_qwen3_4b_8spec")
DATA_DIR = Path(env("DSP_DATA_DIR", str(OUTPUT_DIR / "data")))
VAL_DATA_DIR = env("DSP_VAL_DATA_DIR", "")
# Raw corpus to tokenize when DATA_DIR is not already prepared: a local
# JSON/JSONL path or an "hf:org/dataset" spec, exactly as prepare_data.py takes.
TRAIN_SRC = env("DSP_TRAIN_SRC", "")
MAX_SAMPLES = env("DSP_MAX_SAMPLES", "")
PREP_WORKERS = env("DSP_PREP_WORKERS", "32")

SEQ_LENGTH = env("DSP_SEQ_LENGTH", "8192")
TARGET_LAYER_IDS = env("DSP_TARGET_LAYER_IDS", "1 9 17 25 33").split()
# 9 slots and 9 speculative tokens: DSpark defaults sample_from_anchor to True,
# so slot 0 predicts as well instead of carrying the verified anchor. DFlash2 and
# XPress take the same 9 and draft 8. Distance from the anchor, not the slot
# index, is what makes their position_{k}_acc curves line up.
BLOCK_SIZE = env("DSP_BLOCK_SIZE", "9")
MAX_ANCHORS = env("DSP_MAX_ANCHORS", "512")
NUM_LAYERS = env("DSP_NUM_LAYERS", "5")
MARKOV_RANK = env("DSP_MARKOV_RANK", "256")
MARKOV_HEAD_TYPE = env("DSP_MARKOV_HEAD_TYPE", "vanilla")  # vanilla | gated | rnn
CONFIDENCE_HEAD_ALPHA = env("DSP_CONFIDENCE_HEAD_ALPHA", "1.0")
LOSS_FN = env("DSP_LOSS_FN", '{"ce": 0.1, "tv": 0.9}')
DECAY_GAMMA = env("DSP_DECAY_GAMMA", "4.0")
EPOCHS = env("DSP_EPOCHS", "1")
LR = env("DSP_LR", "6e-4")
WEIGHT_DECAY = env("DSP_WEIGHT_DECAY", "0.0")
WARMUP_RATIO = env("DSP_WARMUP_RATIO", "0.04")
CHECKPOINT_FREQ = env("DSP_CHECKPOINT_FREQ", "0.1")
LOG_FREQ = env("DSP_LOG_FREQ", "50")
RUN_NAME = env("DSP_RUN_NAME", "dspark-qwen3-4b-8spec")
# Stop after N optimizer steps. For a smoke run the question is whether the
# pipeline reaches training at all, and an epoch over millions of samples is a
# very expensive way to answer it.
MAX_STEPS = env("DSP_MAX_STEPS", "")
# Accept length is an eval-only metric: the free-running rollout runs under
# no_grad in val_epoch, never during training. With one epoch and no interval
# it is reported exactly once, at the end -- too late to tell whether the head
# is learning. Counted in optimizer steps, so the cadence is a fixed sample
# interval whatever the world size.
EVAL_INTERVAL = env("DSP_EVAL_INTERVAL", "")
EVAL_MAX_BATCHES = env("DSP_EVAL_MAX_BATCHES", "")
# Resume is right for a long run and wrong for a smoke test: with --epochs 1,
# a checkpoint from an earlier attempt makes the trainer conclude the run is
# already finished and exit in seconds, having trained nothing.
NO_RESUME = env("DSP_NO_RESUME", "")
# Prepare the corpus and exit, without training. Tokenizing runs on rank 0
# alone while the other seven ranks sleep in wait_for_prep, so a long prepare
# inside the training job idles seven GPUs for its whole duration. Run this as
# a one-GPU job first; the training job then finds the stamp and skips it.
PREP_ONLY = env("DSP_PREP_ONLY", "")

VLLM_PORT = int(env("DSP_VLLM_PORT", "8300"))
# Host the trainers and the health check dial. Loopback by default; set it to
# the pod's own address if the container's "lo" is not usable, which shows up
# as every connection to 127.0.0.1 timing out.
VLLM_HOST = env("DSP_VLLM_HOST", "127.0.0.1")
VLLM_VENV = Path(env("DSP_VLLM_VENV", "/gpfs/zwang33/venv_vllm"))
VLLM_PY = Path(env("DSP_VLLM_PY", str(VLLM_VENV / "bin" / "python")))

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
    e["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = env("DSP_NCCL_HEARTBEAT_SEC", "3600")
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
            "dataset_info.json) and DSP_TRAIN_SRC is unset, so there is nothing to "
            "build it from. Online training generates only the HIDDEN STATES on the "
            "fly; the tokenized data still has to exist. Set DSP_TRAIN_SRC to a "
            'JSONL path or an "hf:org/dataset" spec, or point DSP_DATA_DIR at an '
            "already-prepared corpus."
        )
    if not VLLM_PY.exists():
        sys.exit(
            f"[r{RANK}] FATAL - verifier interpreter {VLLM_PY} not found; "
            "set DSP_VLLM_VENV to a venv that has vllm installed."
        )


def serve_vllm_until_done() -> int:
    """Verifier role (LOCAL_RANK 0): serve on GPU 0 until the trainers are done.

    The framework's torchrun waits for every worker, so this process must exit on
    its own once training finishes -- train rank 0 writes DONE for that.
    """
    # The hidden-state spool is the verifier->trainer channel. Leftovers from a
    # crashed run would be read back as if freshly generated (--on-missing generate),
    # silently training on stale activations.
    spool = Path(env("DSP_HIDDEN_STATES_DIR", "/tmp/hidden_states_dspark"))
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
            VLLM_HOST_IP=env("DSP_VLLM_HOST_IP", "127.0.0.1"),
            HOST_IP=env("DSP_VLLM_HOST_IP", "127.0.0.1"),
            # NCCL settings tuned for the trainer fabric do not apply to a single-GPU
            # engine and can send it hunting for IB devices.
            NCCL_IB_DISABLE="1",
            NCCL_P2P_DISABLE="1",
            VLLM_USE_FLASHINFER_SAMPLER="0",
            VLLM_ATTENTION_BACKEND=env("DSP_VLLM_ATTN_BACKEND", "FLASH_ATTN"),
        )
    )
    # Optional, and only applied when non-empty. Everything this launcher adds
    # over the XPress one lives here, so setting them all empty reduces the
    # verifier to that launcher's exact invocation -- which is the only way to
    # separate "our extra flags broke it" from "the environment changed".
    #   DSP_VLLM_V1_MP=0     run EngineCore in-process (no loopback TCPStore)
    #   DSP_VLLM_USE_LIBUV=0 legacy store backend; libuv's listener can fail
    #                        silently in a restricted container
    for _var, _key in (
        ("DSP_VLLM_V1_MP", "VLLM_ENABLE_V1_MULTIPROCESSING"),
        ("DSP_VLLM_USE_LIBUV", "USE_LIBUV"),
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
    # /render templates and tokenizes; it never runs the model, so it is bound by
    # the API server process, not the GPU. More servers on the same GPU is what
    # raises prepare throughput -- more GPUs would do nothing.
    servers = env("DSP_VLLM_API_SERVERS", "")
    if servers:
        cmd += ["--api-server-count", servers]
    bind = env("DSP_VLLM_BIND", "")
    if bind:
        cmd += ["--host", bind]
    # Off by default: the render endpoint validates a request against
    # max_model_len BEFORE prepare_data gets to truncate it to --seq-length, so
    # capping the server at the training length rejects every conversation longer
    # than that ("prompt contains at least 8195 input tokens"). Left unset, vLLM
    # uses the model's own context and truncation happens where it should.
    if env("DSP_VLLM_MAX_MODEL_LEN", "0") == "1":
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
        if PREP_ONLY == "1":
            print(f"[r{RANK}] prep-only: corpus ready at {DATA_DIR}", flush=True)
            return 0
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
    # MODEL belongs in the stamp: input_ids and loss_mask are produced with that
    # model's tokenizer and chat template, so a corpus prepared for one verifier
    # is silently wrong for another. Everything else about a run -- speculator
    # type, block size, losses, learning rate -- is decided at training time and
    # can reuse the same corpus.
    want = f"{MODEL}|{TRAIN_SRC}|{MAX_SAMPLES or 'all'}|{SEQ_LENGTH}"
    # A corpus written before MODEL joined the stamp carries the three-field form.
    # Its bytes are still correct whenever the model matches, and the only way to
    # know that is that the rest of the stamp does -- so accept it, say so, and
    # rewrite the stamp rather than re-tokenizing tens of hours of text.
    legacy = want.split("|", 1)[1]
    if data_is_prepared():
        have = STAMP.read_text() if STAMP.exists() else ""
        if have == legacy:
            print(
                f"[r{RANK}] {DATA_DIR} carries a pre-MODEL stamp; assuming {MODEL} "
                "and upgrading it",
                flush=True,
            )
            STAMP.write_text(want)
            have = want
        if STAMP.exists() and have != want:
            sys.exit(
                f"[r{RANK}] FATAL - {DATA_DIR} was built with different settings.\n"
                f"  want: {want}\n  have: {have}\n"
                "Point DSP_DATA_DIR somewhere else rather than overwriting it."
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


def wait_for_prep(timeout_s: int | None = None) -> None:
    # Tokenizing the full regenerated collection is tens of GB of text; a cap that
    # is too low makes every trainer exit while rank 0 is still working, and the
    # job dies with no indication that it was simply not finished yet.
    timeout_s = timeout_s or int(env("DSP_PREP_TIMEOUT_S", "43200"))
    print(
        f"[r{RANK}] waiting for rank 0 to finish data prep "
        f"(up to {timeout_s // 3600}h)",
        flush=True,
    )
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
        "dspark",
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
        "--markov-rank",
        MARKOV_RANK,
        "--markov-head-type",
        MARKOV_HEAD_TYPE,
        "--enable-confidence-head",
        "--confidence-head-with-markov",
        "--confidence-head-alpha",
        CONFIDENCE_HEAD_ALPHA,
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
    if MAX_STEPS:
        cmd += ["--max-steps", MAX_STEPS]
    if EVAL_INTERVAL:
        cmd += ["--eval-interval", EVAL_INTERVAL]
    if EVAL_MAX_BATCHES:
        cmd += ["--eval-max-batches", EVAL_MAX_BATCHES]
    if NO_RESUME == "1":
        cmd += ["--no-resume-from-checkpoint"]
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
    if PREP_ONLY == "1":
        print(f"[r{RANK}] prep-only: nothing to do on a trainer rank", flush=True)
        return 0
    wait_for_prep()
    return train()


if __name__ == "__main__":
    raise SystemExit(main())
