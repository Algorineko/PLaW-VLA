#!/bin/bash
# Generic DCU training launcher. Usage:
#   bash dcu/train_stage.sh <config_name> [extra train_pytorch.py args...]
# Env overrides: EXP_NAME, NUM_GPUS, CHECKPOINT_DIR, INIT_WEIGHT, EXTRA_ARGS
set -euo pipefail
cd "$(dirname "$0")/.."
source dcu/env.sh

CONFIG="${1:?usage: bash dcu/train_stage.sh <config_name> [extra args...]}"
shift || true

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/checkpoints}"
EXP_NAME="${EXP_NAME:-$CONFIG}"
mkdir -p "$CHECKPOINT_DIR"

ARGS=()
if [ -n "${INIT_WEIGHT:-}" ]; then
  ARGS+=(--pytorch_weight_path "$INIT_WEIGHT")
fi

echo "== launching $CONFIG on $NUM_GPUS DCU(s), exp=$EXP_NAME =="
exec $PLAW_TORCHRUN --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  scripts/train_pytorch.py "$CONFIG" \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CHECKPOINT_DIR" \
  ${EXTRA_ARGS:-} \
  "$@"
