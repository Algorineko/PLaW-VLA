#!/bin/bash
# Hygon DCU (BW1000_H / DTK-25.04) environment for PLaW-VLA on this box.
# Usage: source dcu/env.sh
#
# Box facts this encodes (see roadmap/output/hygon-dcu-training/SKILL.md):
# - 2x BW1000_H 64GiB; select cards with HIP_VISIBLE_DEVICES (NOT CUDA_VISIBLE_DEVICES).
# - Host cgroup memory hard wall = 32GiB; hy-smi HCU% is fake 0.0, trust VRAM%.
# - DeepSpeed/ZeRO unusable on the das torch build; plain DDP only.
# - hf_transfer crashes on this box; always use the mirror + disable it.

set -euo pipefail

export HCU_ROOT="${HCU_ROOT:-/home/tione/notebook/home/arianliu/env/hcu_root}"
export PLAW_VENV="${PLAW_VENV:-/home/tione/notebook/home/arianliu/venvs/plaw-venv}"

# NOTE: unlike bare-vendor-python activation, the venv recipe must NOT set
# PYTHONHOME/PYTHONPATH -- they hijack venv module resolution (pip installs
# land in the wrong tree). The venv sees das torch via --system-site-packages.
unset PYTHONHOME PYTHONPATH || true

# DTK runtime libraries.
set +u
# shellcheck disable=SC1091
[ -f "$HCU_ROOT/opt/dtk-25.04/env.sh" ] && source "$HCU_ROOT/opt/dtk-25.04/env.sh"
set -u

PYTHON="$PLAW_VENV/bin/python"
export PYTHON
# Never invoke the bare `torchrun` console script (its shebang points at the
# system python); go through the interpreter explicitly.
PLAW_TORCHRUN="$PYTHON -m torch.distributed.run"
export PLAW_TORCHRUN

# GPU selection (HIP, not CUDA).
# 2026-10-07 规定：复现工作只用 0 卡；1 卡留给其他项目（恢复双卡另行通知）。
export SAFI_GPU="${SAFI_GPU:-0}"
export HIP_VISIBLE_DEVICES="$SAFI_GPU"

# Download hygiene.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=0

# Trainer behavior on DCU.
export JAX_PLATFORMS=cpu                 # JAX is tree-utils only; keep it off the DCU
export WANDB_MODE="${WANDB_MODE:-offline}"
export PLAW_VLA_TORCH_COMPILE_MODE="${PLAW_VLA_TORCH_COMPILE_MODE:-off}"  # requires dcu-branch patch
export PLAW_VLA_ADAMW_FOREACH="${PLAW_VLA_ADAMW_FOREACH:-0}"              # foreach transients OOM the 64G card
export PLAW_VLA_SAVE_LOCK="${PLAW_VLA_SAVE_LOCK:-$PWD/.save_lock}"        # serialize ckpt saves vs the 32G wall
export PLAW_VLA_INIT_ON_DEVICE="${PLAW_VLA_INIT_ON_DEVICE:-1}"    # GPU-context model init (host 32G wall, runbook B2)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128                      # expandable_segments unsupported on das

# torch extension visibility for distributed runs.
# 单卡跑批时保持配方等效 batch 用 --grad-accum-steps 补（如 stage3: --batch-size 64 --grad-accum-steps 4）。
export NUM_GPUS="${NUM_GPUS:-1}"

echo "dcu env ready: python=$PYTHON gpus=$SAFI_GPU venv=$PLAW_VENV"
