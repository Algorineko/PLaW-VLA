#!/bin/bash
# Pre-flight before any PLaW-VLA DCU training. Run: bash dcu/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source dcu/env.sh

echo "== 1. cgroup host-RAM quota (THE wall) =="
LIM=/sys/fs/cgroup/memory/memory.limit_in_bytes
if [ -r "$LIM" ]; then
  V=$(cat "$LIM"); echo "limit = $((V/1024/1024/1024))GB, usage = $(awk '{print $1/1024/1024/1024 " GB"}' /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)"
else
  cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/memory.current 2>/dev/null
fi

echo; echo "== 2. GPU state (trust VRAM%, HCU% is fake 0.0) =="
/opt/hyhal/bin/hy-smi | sed -n '1,3p;/^[0-9] /p'

echo; echo "== 3. venv stack =="
"$PYTHON" - <<'EOF'
import torch, transformers, jax, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
print("nccl(rccl)", torch.distributed.is_nccl_available())
print("transformers", transformers.__version__, "| jax", jax.__version__, "| numpy", numpy.__version__)
import openpi  # repo editable install or PYTHONPATH
print("openpi import OK from", openpi.__file__)
EOF

echo; echo "== 4. assets present =="
for p in data/libero_v3_eef/meta/info.json \
         checkpoints/pi05_base_pytorch/model.safetensors \
         assets/stage3_finetuning_libero/libero_v3_eef/norm_stats.json \
         assets/stage2_pretraining/libero_v3_eef/norm_stats.json; do
  [ -e "$p" ] && echo "  OK   $p" || echo "  MISS $p"
done
python - <<'EOF'
import os, glob
home = os.path.expanduser("~")
for pat in ("~/.cache/huggingface/hub/models--facebook--vjepa2*", "~/.cache/plaw-vla/*"):
    hits = glob.glob(os.path.join(home, pat[2:])) or glob.glob(pat.replace("~", os.path.expanduser("~")))
    print("  OK  ", hits[:1] if hits else "  MISS " + pat)
EOF

echo; echo "== 5. top host-RAM consumers =="
ps -eo rss,pid,comm --sort=-rss | head -6
echo "preflight done."
