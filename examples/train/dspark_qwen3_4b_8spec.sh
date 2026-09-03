#!/bin/bash
# DSpark Training Script — Qwen3-4B — 8 Speculative Tokens
#
# The DFlash2 recipe (examples/train/dflash2_qwen3_4b_8spec.sh) with DSpark's two
# heads in place of DFlash2's convolution and selector: a low-rank Markov head
# that conditions each draft position on the token before it, and a confidence
# head that predicts per-position acceptance. Everything else — backbone size,
# block layout, optimizer, schedule, data — is held equal so the two runs are
# comparable.
#
# block_size 9 with sample_from_anchor=False yields 8 speculative tokens.
#
# Prerequisites:
#   - 8x GPUs (1 for vLLM, 7 for training)
#   - A prepared corpus at DATA_PATH (scripts/prepare_data.py output). Online
#     training generates only the hidden states on the fly; the tokenized data
#     still has to exist.
#
# Usage:
#   bash examples/train/dspark_qwen3_4b_8spec.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# ============ Logging ============
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_qwen3_4b_8spec}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# ============ Configuration ============
MODEL="Qwen/Qwen3-4B"
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data/Qwen3-4B-preprocessed}"
VLLM_PORT=8300
SEQ_LENGTH=8192
EPOCHS=1
LR=6e-4

SPECULATOR_TYPE="dspark"
BLOCK_SIZE=9              # 9 − 1 = 8 speculative tokens (sample_from_anchor=False)
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"   # Qwen3-4B has 36 layers

# DSpark heads
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"        # vanilla | gated | rnn
CONFIDENCE_HEAD_ALPHA=1.0

# Optimizer / loss — identical to the DFlash2 recipe
OPTIMIZER="adamw"
WEIGHT_DECAY=0.0
SCHEDULER_TYPE="cosine"
SCHEDULER_WARMUP_RATIO=0.04
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
DFLASH_DECAY_GAMMA=4.0

# One GPU serves the verifier; a single server saturates long before a second
# card is needed, so the rest train.
VLLM_GPUS="0"
TRAIN_GPUS="1,2,3,4,5,6,7"
NUM_TRAIN_GPUS=7

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "*** DRY RUN: will build model, save config, and exit ***"
fi
# =======================================

if [[ ! -f "$DATA_PATH/dataset_info.json" ]]; then
    echo "FATAL: $DATA_PATH is not prepare_data.py output (no dataset_info.json)."
    echo "Build it with scripts/prepare_data.py, or set DATA_PATH."
    exit 1
fi

# ============ Step 1: Launch vLLM Server ============
echo "=== Step 1: Launching vLLM server ==="
# No --max-model-len: the render endpoint validates a request against it before
# any truncation happens, so capping it at the training length rejects every
# longer conversation.
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# ============ Step 2: Train DSpark ============
echo "=== Step 2: Training DSpark speculator (8 speculative tokens) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --speculator-type "$SPECULATOR_TYPE" \
    --data-path "$DATA_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --dflash-decay-gamma "$DFLASH_DECAY_GAMMA" \
    --per-position-loss-weight fixed-exp-decay \
    --optimizer "$OPTIMIZER" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --scheduler-type "$SCHEDULER_TYPE" \
    --scheduler-warmup-ratio "$SCHEDULER_WARMUP_RATIO" \
    --epochs "$EPOCHS" \
    --total-seq-len "$SEQ_LENGTH" \
    --seed 42 \
    --fsdp-shard \
    --on-missing generate \
    --on-generate delete \
    --checkpoint-freq 0.1 \
    $DRY_RUN

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
