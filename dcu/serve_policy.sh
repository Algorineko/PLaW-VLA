#!/bin/bash
# Serve a trained checkpoint on a DCU. Usage:
#   bash dcu/serve_policy.sh <checkpoint_step_dir> [config_name] [gpu_id]
set -euo pipefail
cd "$(dirname "$0")/.."
source dcu/env.sh

CKPT_DIR="${1:?usage: bash dcu/serve_policy.sh <checkpoint_step_dir> [config_name] [gpu_id]}"
CONFIG_NAME="${2:-stage3_finetuning_libero}"
GPU="${3:-0}"

SAFI_GPU="$GPU" HIP_VISIBLE_DEVICES="$GPU" exec "$PYTHON" scripts/serve_policy.py \
  --env LIBERO \
  policy:checkpoint \
  --policy.config="$CONFIG_NAME" \
  --policy.dir="$CKPT_DIR"
