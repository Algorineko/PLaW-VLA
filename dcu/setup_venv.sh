#!/bin/bash
# Build the DCU venv for PLaW-VLA (P2 path) from scratch — the recipe that
# passed full verification on 2026-10-07.
#
#   torch 2.4.1+das (system, via --system-site-packages) + torchvision 0.19.1+das
#   transformers 5.0.0 + huggingface-hub>=1.0 + lerobot@017ff73f (--no-deps)
#   numpy PINNED to 1.26.4 (numpy>=2 breaks das torch CUDA init!)
#
# Re-run transformers patch after this: see dcu/README.md.
set -euo pipefail
export HCU_ROOT="${HCU_ROOT:-/home/tione/notebook/home/arianliu/env/hcu_root}"
unset PYTHONHOME PYTHONPATH
set +u; source "$HCU_ROOT/opt/dtk-25.04/env.sh"; set -u

VENV="${PLAW_VENV:-/home/tione/notebook/home/arianliu/venvs/plaw-venv}"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"   # pypi.org is 2s-RTT slow here

if [ ! -x "$VENV/bin/python" ]; then
  "$HCU_ROOT/usr/bin/python3.10" -m venv --without-pip --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install -q --force-reinstall pip
fi
VPY="$VENV/bin/python"

echo "== 1. transformers stack (deps included) =="
$VPY -m pip install -i "$MIRROR" "transformers==5.0.0" "huggingface-hub>=1.0,<2" "safetensors>=0.4" "tokenizers>=0.20" \
  "numpy==1.26.4"   # keep the pin EXPLICIT: several deps try to pull numpy>=2

echo "== 2. jax-cpu tree utils + trainer deps =="
$VPY -m pip install -i "$MIRROR" jax==0.5.3 jaxlib==0.5.3 flax==0.10.2 orbax-checkpoint==0.11.13 \
  ml_collections equinox augmax jaxtyping==0.2.36 beartype==0.19.0 etils treescope \
  tyro tqdm-loggable wandb rich filelock accelerate "av>=15" opencv-python==4.10.0.84 \
  imageio pillow sentencepiece timm polars numpydantic "fsspec[gcs]" graphql-core \
  "datasets>=4.0.0,<5.0.0" pyarrow "numpy==1.26.4"

echo "== 3. lerobot@017ff73f (--no-deps: its torch/torchvision/torchcodec would shadow das torch) =="
LEROBOT_REV=017ff73fbfe46bf9a673cd9b402988dcb79151f7
SRC="/tmp/lerobot-$LEROBOT_REV"
if [ ! -d "$SRC" ]; then
  mkdir -p "$SRC"
  curl -sL -o /tmp/lerobot.tar.gz "https://codeload.github.com/huggingface/lerobot/tar.gz/$LEROBOT_REV"
  tar -xzf /tmp/lerobot.tar.gz -C /tmp
fi
$VPY -m pip install -i "$MIRROR" --no-deps --ignore-requires-python "$SRC"
# lerobot runtime deps the data path actually imports (video_utils: av/fsspec/numpy/pyarrow/torch/torchvision):
$VPY -m pip install -i "$MIRROR" --no-deps "opencv-python==4.10.0.84"

echo "== 4. repo editable install (py>=3.12 pin bypassed) =="
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
$VPY -m pip install -e "$REPO" --no-deps --ignore-requires-python

echo "== 5. transformers file patch =="
FORCE_SYNC=0 PYTHON_BIN="$VPY" bash "$REPO/scripts/install_transformers_patch.sh"

echo "== 6. purge orphan dist-info that lies about torch versions =="
VSP="$($VPY -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "$VSP"/torch-*.dist-info "$VSP"/torchcodec-*.dist-info "$VSP"/torchvision-*.dist-info
# (torch itself stays: it lives in hcu_root system site-packages)

echo "== 7. verify =="
HIP_VISIBLE_DEVICES="${SAFI_GPU:-0}" "$VPY" - <<'EOF'
import numpy, torch, torchvision, transformers, jax, lerobot
print("numpy", numpy.__version__, "| torch", torch.__version__, "| tv", torchvision.__version__)
assert torch.cuda.is_available(), "das torch cannot see the DCU"
x = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"); x @ x
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), "transformers patch probe failed"
import openpi
print("ALL GREEN: cuda", torch.cuda.device_count(), "dev(s); openpi at", openpi.__file__)
EOF
echo VENV_SETUP_DONE
