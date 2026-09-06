#!/usr/bin/env python
"""Multi-pod launcher for XPress training on the k8s cluster.

Layout per pod: GPU 0 runs a vLLM verifier (serving target hidden states over
localhost), GPUs 1-7 run 7 trainer ranks. With 4 pods that is 4 verifiers and
28 trainer ranks -- hidden states never cross the pod boundary.

The k8s job framework injects RANK (pod index), WORLD_SIZE (pod count),
MASTER_ADDR and MASTER_PORT; this script turns those into a torchrun rendezvous
and serializes the one-time prep so only pod 0 writes to the shared volume.

Everything is idempotent: re-running a failed job reuses the converted backbone
and the prepared data.
"""

import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


# The job framework launches `torchrun --nnodes=N --nproc_per_node=<gpus>` over this
# script, so every process here owns ONE GPU: RANK is global (0..31), LOCAL_RANK is
# the GPU index on its node, GROUP_RANK is the node index.
RANK = int(env("RANK", "0"))
WORLD = int(env("WORLD_SIZE", "1"))
LOCAL_RANK = int(env("LOCAL_RANK", "0"))
LOCAL_WORLD = int(env("LOCAL_WORLD_SIZE", "1"))
NODE = int(env("GROUP_RANK", "0"))
NUM_NODES = max(1, WORLD // max(1, LOCAL_WORLD))
MASTER_ADDR = env("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = int(env("MASTER_PORT", "23456"))

# GPU 0 of each node serves the vLLM verifier; the rest train. The trainers form
# their OWN process group (the framework's rendezvous includes the verifiers), so
# they get a remapped rank and a port of their own.
IS_VERIFIER = LOCAL_RANK == 0
TRAINERS_PER_NODE = max(1, LOCAL_WORLD - 1)
TRAIN_WORLD = NUM_NODES * TRAINERS_PER_NODE
TRAIN_RANK = NODE * TRAINERS_PER_NODE + (LOCAL_RANK - 1)
TRAIN_PORT = int(env("XP_TRAIN_PORT", str(MASTER_PORT + 1)))

MODEL = env("XP_MODEL", "Qwen/Qwen3-8B")
OUT_ROOT = Path(env("XP_OUT_ROOT", "/gpfs/zwang33/xpress"))
BACKBONE = OUT_ROOT / "dflash_b16_zlab_converted"
OUTPUT_DIR = OUT_ROOT / env("XP_RUN_DIR", "xpress_b16_32gpu")
# Tokenised corpora are setting-independent and expensive to build, so A/B runs
# should SHARE one: point XP_DATA_DIR at an existing run's directory and prepare()
# no-ops on the per-split .data_ready stamp. Defaults to this run's own directory.
DATA_DIR = Path(env("XP_DATA_DIR", str(OUTPUT_DIR)))
TRAIN_DATA = DATA_DIR / "train"
TRAIN_JSONL = Path(env("XP_TRAIN_JSONL", "/gpfs/zwang33/dflash_data/refiner_train_nothink.jsonl"))

SEQ_LENGTH = int(env("XP_SEQ_LENGTH", "4096"))
TARGET_LAYER_IDS = env("XP_TARGET_LAYER_IDS", "1 9 17 25 33").split()
VLLM_PORT = int(env("XP_VLLM_PORT", "8300"))
# The verifier gets its OWN interpreter: vLLM and the trainer disagree on
# transformers/torch pins, so they cannot share one environment (the single-node
# recipe uses .venv_vllm for the same reason). Built once on the shared volume.
VLLM_VENV = Path(env("XP_VLLM_VENV", "/gpfs/zwang33/venv_vllm"))
VLLM_PY = Path(env("XP_VLLM_PY", str(VLLM_VENV / "bin" / "python")))
VLLM_SPEC = env("XP_VLLM_SPEC", "vllm>=0.22.0")
MAX_SAMPLES = env("XP_MAX_SAMPLES", "1311126")
LOG_FREQ = env("XP_LOG_FREQ", "50")
RUN_NAME = env("XP_RUN_NAME", "xpress-b16-32gpu")
READY = OUTPUT_DIR / ".prep_ready"
DONE = OUTPUT_DIR / ".train_done"


# Everything the framework's torchrun injects to describe the 32-process job.
# A child that is NOT part of that job must not see them.
_DIST_VARS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
              "GROUP_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
              # torchrun also sets TORCHELASTIC_USE_AGENT_STORE=True, and torch's TCP
              # rendezvous reads it: with it set, init_process_group builds a CLIENT
              # store and waits for the agent to be listening -- even at world_size=1,
              # where rank 0 would otherwise create the server itself. A child that is
              # not part of the job then hangs for the full store timeout (10 min) on a
              # listener that will never exist. Observed on vLLM 0.27.1, whose EngineCore
              # calls init_process_group in a subprocess; earlier versions did not, which
              # is why this sat here harmlessly until an upgrade set it off.
              "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_RESTART_COUNT",
              "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
              "TORCHELASTIC_ERROR_FILE", "TORCH_ELASTIC_WORKER_IDENTIFIER")


def strip_dist(e: dict) -> dict:
    for k in list(e):
        if k in _DIST_VARS or k.startswith(("TORCHELASTIC_", "ROLE_")):
            e.pop(k, None)
    return e


def child_env(**extra: str) -> dict:
    """Environment for every subprocess.

    PYTHONPATH carries the repo's packages explicitly: `pip install -e .` in the
    job's setup step has landed in a different interpreter than the one running
    this launcher before, and the failure only surfaced deep in a child process.
    Adding the source roots makes the children importable either way.
    """
    e = dict(os.environ)
    roots = [str(REPO / "src"), str(REPO / "hs_connectors" / "src")]
    e["PYTHONPATH"] = os.pathsep.join(roots + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []))

    # Give each NODE its own compile cache. The job spec points these at shared /gpfs,
    # where inductor's write-then-rename is not atomic across nodes: a rank can dlopen
    # a .so another node is still writing and die with "file too short". A per-host
    # subdirectory removes that race yet keeps the cache persistent, which matters --
    # a cold cache cost ~8 minutes of compilation, and see the heartbeat note below.
    for _var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        _root = e.get(_var)
        if _root:
            _d = os.path.join(_root, socket.gethostname())
            os.makedirs(_d, exist_ok=True)
            e[_var] = _d

    # Inductor holds the GIL for the whole compile, so NCCL's watchdog thread cannot
    # tick; the heartbeat monitor reads that as a wedged rank and aborts the job
    # ("watchdog got stuck for N seconds"). Set it HERE rather than trusting the pod
    # spec: a run died at PyTorch's 480s default while the job yaml asked for 1800.
    e["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = env("XP_NCCL_HEARTBEAT_SEC", "3600")

    e.update(extra)
    return e


def run(cmd: list[str], **kw) -> None:
    print(f"[r{RANK}] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    kw.setdefault("env", child_env())
    subprocess.run(cmd, check=True, **kw)


def vllm_env_ok() -> bool:
    """True if VLLM_PY can import vllm AND the hidden-state connector is registered."""
    if not VLLM_PY.exists():
        return False
    probe = (
        "from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory as F;"
        "import hs_connectors, sys;"
        "sys.exit(0 if 'ExampleHiddenStatesConnector' in getattr(F, '_registry', {}) else 1)"
    )
    return subprocess.run([str(VLLM_PY), "-c", probe],
                          capture_output=True).returncode == 0


def fix_flashinfer() -> None:
    """Drop flashinfer from the verifier venv if it cannot be imported.

    flashinfer's fd_exchange module annotates `array.array[int]`, which only parses
    on Python 3.12+; on 3.11 it raises TypeError at import. vLLM guards that import
    against ImportError only, so the TypeError propagates and kills EngineCore.
    Without the package vLLM falls back to its native sampling kernels, which is
    fine here -- the verifier only produces hidden states.
    """
    probe = "import flashinfer.comm"
    if subprocess.run([str(VLLM_PY), "-c", probe], capture_output=True).returncode == 0:
        return
    print(f"[r{RANK}] flashinfer unusable on this interpreter; removing it from "
          f"{VLLM_VENV}", flush=True)
    subprocess.run([str(VLLM_VENV / "bin" / "pip"), "uninstall", "-y",
                    "flashinfer", "flashinfer-python"], capture_output=True)


def ensure_vllm_env() -> None:
    """Pod 0 only: build the verifier venv if it is missing or incomplete."""
    if vllm_env_ok():
        print(f"[r{RANK}] verifier env ready: {VLLM_PY}", flush=True)
        fix_flashinfer()
        return
    print(f"[r{RANK}] building verifier venv at {VLLM_VENV} (one-time, ~10 min)", flush=True)
    if not VLLM_PY.exists():
        # The validated recipe builds this venv on 3.12; several vLLM extras (e.g.
        # flashinfer) use 3.12-only annotations. Fall back to our own interpreter
        # when 3.12 is absent -- fix_flashinfer() then prunes what cannot load.
        base = shutil.which("python3.12") or sys.executable
        print(f"[r{RANK}] creating verifier venv with {base}", flush=True)
        run([base, "-m", "venv", str(VLLM_VENV)])
    pip = [str(VLLM_VENV / "bin" / "pip"), "install",
           "--cache-dir", env("PIP_CACHE_DIR", "/gpfs/zwang33/cache/pip")]
    run(pip + [VLLM_SPEC])
    run(pip + ["-e", "./hs_connectors"])       # registers ExampleHiddenStatesConnector
    if not vllm_env_ok():
        sys.exit(f"[r{RANK}] FATAL: {VLLM_PY} still lacks vllm or the "
                 "ExampleHiddenStatesConnector after install")
    fix_flashinfer()


def prepare() -> None:
    """Pod 0 only: convert the backbone, morph its config, tokenize the corpus."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_vllm_env()
    if not (BACKBONE / "config.json").exists():
        print(f"[r{RANK}] converting z-lab b16 -> speculators format", flush=True)
        run([sys.executable, "-c", (
            "from speculators.convert.entrypoints import convert_model;"
            f"convert_model(model='z-lab/Qwen3-8B-DFlash-b16', verifier='{MODEL}',"
            f" algorithm='dflash', output_path='{BACKBONE}',"
            " aux_hidden_state_layer_ids=[1, 9, 17, 25, 33])")])
    run([sys.executable, "examples/train/xpress_morph_config.py", str(BACKBONE),
         "--rank", "256", "--mlp-ratio", "2"])

    for src, dst, cap in ((TRAIN_JSONL, TRAIN_DATA, MAX_SAMPLES),):
        stamp = dst / ".data_ready"
        want = f"{src}|{cap or 'all'}|{SEQ_LENGTH}|{src.stat().st_size}"
        if stamp.exists() and stamp.read_text() == want:
            print(f"[r{RANK}] [skip] {dst} already prepared", flush=True)
            continue
        if DATA_DIR != OUTPUT_DIR:
            # Borrowed corpus (XP_DATA_DIR points at another run). Rebuilding would
            # run prepare_data --overwrite on data that run owns, and it would then
            # resume onto a silently different corpus. Refuse instead: whatever
            # differs here is a real disagreement about what the data should be.
            raise SystemExit(
                f"FATAL - {dst} is a BORROWED corpus (XP_DATA_DIR) and does not match "
                f"this run's settings, so preparing it would overwrite another run's "
                f"data.\n  want: {want}\n  have: "
                f"{stamp.read_text() if stamp.exists() else '<no .data_ready stamp>'}\n"
                f"Either point XP_DATA_DIR elsewhere, or unset it to build a private copy."
            )
        stamp.unlink(missing_ok=True)
        cmd = [sys.executable, "scripts/prepare_data.py", "--model", MODEL,
               "--data", str(src), "--output", str(dst),
               "--seq-length", str(SEQ_LENGTH), "--overwrite"]
        if cap:
            cmd += ["--max-samples", cap]
        run(cmd)
        stamp.write_text(want)
    READY.write_text("ok")


def wait_for_prep(timeout_s: int = 7200) -> None:
    print(f"[r{RANK}] waiting for pod 0 to finish data prep", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if READY.exists():
            print(f"[r{RANK}] prep ready", flush=True)
            return
        time.sleep(15)
    sys.exit(f"[r{RANK}] FATAL: pod 0 did not finish prep within {timeout_s}s")


def vllm_healthy(timeout_s: float = 5.0) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/v1/models", timeout=timeout_s)
        return True
    except Exception:
        return False


def wait_for_local_vllm(timeout_s: int = 2400) -> None:
    """Trainer role: block until THIS node's verifier answers.

    The prep marker only means the data is ready; the verifier still has to load
    an 8B model afterwards. Without this wait the dataloader's first request hits
    a closed port and the rank dies with ConnectionRefused.
    """
    if vllm_healthy():
        return
    print(f"[r{RANK}] waiting for node {NODE}'s verifier on port {VLLM_PORT}", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if vllm_healthy():
            print(f"[r{RANK}] verifier reachable, starting training", flush=True)
            return
        time.sleep(10)
    sys.exit(f"[r{RANK}] FATAL: node {NODE}'s verifier not reachable within {timeout_s}s")


def serve_vllm_until_done() -> int:
    """Verifier role (LOCAL_RANK 0): serve on GPU 0 until the trainers are done.

    The framework's torchrun waits for every worker, so these processes must exit
    on their own once training finishes -- train rank 0 writes DONE for that.
    A trainer crash instead makes torchrun tear the whole job down.
    """
    if not vllm_env_ok():
        sys.exit(f"[r{RANK}] FATAL: verifier env {VLLM_PY} unusable "
                 "(rank 0 should have built it before releasing the prep marker)")
    # Clear this node's hidden-state spool. It is the verifier->trainer channel and
    # lives in the container's /tmp; leftovers from a crashed run would be read back
    # as if freshly generated (--on-missing generate), silently training on stale
    # activations. Safe here: the trainers only start once we answer health checks.
    spool = Path(env("XP_HIDDEN_STATES_DIR", "/tmp/hidden_states"))
    if spool.exists():
        print(f"[r{RANK}] clearing stale spool {spool}", flush=True)
        shutil.rmtree(spool, ignore_errors=True)

    # vLLM runs its OWN single-GPU process group. Inheriting the job's rendezvous
    # vars makes it try to join the 32-member group instead -- it then waits for
    # ranks that are themselves waiting for it to come up.
    e = strip_dist(child_env(
        CUDA_VISIBLE_DEVICES="0",
        # The verifier is one process on one GPU, so its internal store belongs on
        # loopback. Left to guess, vLLM picks the pod's routable IP and its own
        # EngineCore then cannot dial back to it (10-minute TCPStore timeout).
        VLLM_HOST_IP=env("XP_VLLM_HOST_IP", "127.0.0.1"),
        HOST_IP=env("XP_VLLM_HOST_IP", "127.0.0.1"),
        # NCCL settings tuned for the 28-rank training fabric do not apply to a
        # single-GPU engine and can send it hunting for IB devices.
        NCCL_IB_DISABLE="1",
        NCCL_P2P_DISABLE="1",
        # flashinfer cannot be imported on this venv's Python (see fix_flashinfer).
        # Two code paths reach it: the compile pass guards with find_spec, so removing
        # the package is enough there, but the sampler imports it unconditionally
        # unless this switch is off. Pinning the attention backend keeps backend
        # selection away from it too. vLLM then uses its native kernels.
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_ATTENTION_BACKEND=env("XP_VLLM_ATTN_BACKEND", "FLASH_ATTN"),
    ))
    cmd = [str(VLLM_PY), "scripts/launch_vllm.py", MODEL,
           "--target-layer-ids", *TARGET_LAYER_IDS, "--port", str(VLLM_PORT)]
    print(f"[r{RANK}] node {NODE}: verifier on GPU 0 ({VLLM_PY})", flush=True)
    proc = subprocess.Popen(cmd, env=e)
    for i in range(360):                                   # up to 30 min
        if i and i % 24 == 0:                              # a heartbeat every 2 min
            print(f"[r{RANK}] node {NODE}: vLLM still starting ({i * 5}s)", flush=True)
        if proc.poll() is not None:
            sys.exit(f"[r{RANK}] FATAL: vLLM exited during startup ({proc.returncode})")
        if vllm_healthy():
            break
        time.sleep(5)
    else:
        proc.kill()
        sys.exit(f"[r{RANK}] FATAL: vLLM not ready in time")
    print(f"[r{RANK}] node {NODE}: vLLM ready, serving until training completes", flush=True)
    try:
        while not DONE.exists():
            if proc.poll() is not None:
                sys.exit(f"[r{RANK}] FATAL: vLLM died while serving ({proc.returncode})")
            time.sleep(30)
        print(f"[r{RANK}] node {NODE}: training done, stopping verifier", flush=True)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


def train() -> int:
    """Trainer role (LOCAL_RANK >= 1): become one rank of the 28-way trainer group.

    LOCAL_RANK doubles as the device index in speculators (set_device_index /
    .to(local_rank)), so keeping the physical GPU number here puts this rank on its
    own GPU and leaves GPU 0 to the verifier.
    """
    # Strip the framework's rendezvous first, then describe OUR 28-rank group.
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
        sys.executable, "scripts/train.py",
        "--verifier-name-or-path", MODEL,
        "--from-pretrained", str(BACKBONE),
        "--data-path", str(TRAIN_DATA),
        "--vllm-endpoint", f"http://localhost:{VLLM_PORT}/v1",
        "--save-path", str(OUTPUT_DIR / "checkpoints"),
        "--epochs", "10",
        "--lr", "6e-4",
        "--scheduler-type", "cosine",
        "--scheduler-warmup-ratio", "0.04",
        "--optimizer", "adamw",
        "--weight-decay", "0.0",
        "--checkpoint-freq", "0.02",
        "--total-seq-len", str(SEQ_LENGTH),
        "--speculator-type", "xpress",
        "--max-anchors", "400",
        "--target-layer-ids", *TARGET_LAYER_IDS,
        "--xpress-rank", "256",
        "--consistency-weight", "0.3",
        "--consistency-passes", "3",
        "--base-anchor-weight", "0.6",
        "--base-anchor-floor", "0.2",
        "--decayed-loss-norm",
        "--log-freq", LOG_FREQ,
        "--ce-from-data",
        "--loss-fn", '{"ce": 0.1, "tv": 1.8}',
        "--on-missing", "generate",
        "--on-generate", "delete",
        "--logger", "wandb",
        "--run-name", RUN_NAME,
    ]
    if TRAIN_RANK == 0:
        print(f"[r{RANK}] trainer group: {TRAIN_WORLD} ranks "
              f"({NUM_NODES} nodes x {TRAINERS_PER_NODE}), "
              f"rdzv {MASTER_ADDR}:{TRAIN_PORT}", flush=True)
        print(f"[r{RANK}] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    print(f"[r{RANK}] node {NODE} gpu {LOCAL_RANK} -> train rank "
          f"{TRAIN_RANK}/{TRAIN_WORLD}", flush=True)
    wait_for_local_vllm()
    rc = subprocess.run(cmd, env=e).returncode
    if TRAIN_RANK == 0 and rc == 0:
        DONE.write_text("ok")          # releases this job's verifiers
    return rc


def diagnose() -> None:
    """One line per process describing what the job framework actually handed us.

    Settles the two questions the orchestration hinges on -- whether the injected
    ranks are pod-level or GPU-level, and how many GPUs a process can see -- without
    needing a dedicated debug job.
    """
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES")
    dump = {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}
    try:
        import torch
        ngpu, tver = torch.cuda.device_count(), torch.__version__
    except Exception as exc:
        ngpu, tver = f"<torch failed: {exc}>", "?"
    spec, detail = "yes", ""
    try:
        import speculators  # noqa: F401
    except Exception as exc:
        # Print the WHOLE failure: a bare exception name hid an OSError (a compiled
        # extension built against the image's older torch) behind what looked like a
        # plain missing module, and PYTHONPATH cannot fix that one.
        import traceback
        spec = f"no ({type(exc).__name__}: {exc})"
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1200:]
    print(f"[r{RANK}] DIAG host={socket.gethostname()} torch={tver} visible_gpus={ngpu} "
          f"speculators={spec} python={sys.executable}", flush=True)
    print(f"[r{RANK}] DIAG env={dump}", flush=True)
    if detail and RANK == 0:
        print(f"[r{RANK}] DIAG speculators import traceback:\n{detail}", flush=True)


def main() -> int:
    diagnose()
    if os.environ.get("XP_DIAG") == "1":
        return 0
    print(f"[r{RANK}] node {NODE}/{NUM_NODES} gpu {LOCAL_RANK} "
          f"role={'verifier' if IS_VERIFIER else f'trainer#{TRAIN_RANK}'}", flush=True)
    if RANK == 0:
        DONE.unlink(missing_ok=True)   # a rerun must not inherit a stale DONE
        prepare()                      # writes READY, releasing the other ranks
    else:
        wait_for_prep()
    return serve_vllm_until_done() if IS_VERIFIER else train()


if __name__ == "__main__":
    raise SystemExit(main())
