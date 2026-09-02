#!/bin/bash
# DFlash2 Training Script — Qwen3-4B — 8 Speculative Tokens
#
# Trains a DFlash2 speculator with block_size=9, yielding 8 speculative tokens
# (sample_from_anchor=False ⇒ speculative_tokens = block_size − 1 = 8).
#
# Uses preprocessed data at /home/shanjiaz/Qwen3-4B-preprocessed.
#
# Prerequisites:
#   - 8x NVIDIA GPUs (4 for vLLM, 4 for training)
#   - Preprocessed data at DATA_PATH (see scripts/prepare_data.py; override
#     with DATA_PATH=... when it lives elsewhere)
#
# Usage:
#   bash examples/train/dflash2_qwen3_4b_8spec.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# ============ Logging ============
OUTPUT_DIR="./output/dflash2_qwen3_4b_8spec"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# ============ Configuration ============
MODEL="Qwen/Qwen3-4B"
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data/Qwen3-4B-preprocessed}"
VLLM_PORT=8000
SEQ_LENGTH=8192
EPOCHS=1
LR=6e-4

# DFlash2-specific parameters
SPECULATOR_TYPE="dflash2"
BLOCK_SIZE=9              # 9 − 1 = 8 speculative tokens (sample_from_anchor=False)
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"

# DFlash2 convolution & selector
CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16
SELECTOR_RANK=256
SELECTOR_TOP_K=16

# Optimizer
OPTIMIZER="adamw"
WEIGHT_DECAY=0.0
SCHEDULER_TYPE="cosine"
SCHEDULER_WARMUP_RATIO=0.04

# Loss
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
DFLASH_DECAY_GAMMA=4.0

# GPU assignments — 4 GPUs for vLLM (DP=4), 4 for training
VLLM_GPUS="0,1,2,3"
VLLM_DP_SIZE=4
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4

# Pass --dry-run as first arg to build model + save config without training
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "*** DRY RUN: will build model, save config, and exit ***"
fi
# =======================================

# ============ Step 1: Launch vLLM Server ============
echo "=== Step 1: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" \
       --data-parallel-size "$VLLM_DP_SIZE" \
       --max-model-len $((SEQ_LENGTH + 2)) &
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

# ============ Step 2: Train DFlash2 ============
echo "=== Step 2: Training DFlash2 speculator (8 speculative tokens) ==="
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
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
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
