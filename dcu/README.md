# DCU (海光 BW1000_H) 运行目录

本目录是 PLaW-VLA 在海光 DCU 上的适配与运维脚本集合，位于 `dcu` 分支。
上游通用修复在 `audit-fixes` 分支；`git diff audit-fixes..dcu` 即全部 DCU 增量。

## 分支内容（source 补丁）

| 补丁 | 位置 | 作用 |
|---|---|---|
| torch.compile 关断 | `src/openpi/models_pytorch/pi0_pytorch.py` | `PLAW_VLA_TORCH_COMPILE_MODE=off` 时跳过编译（上游该 env 只改 mode 值，非法值直接报错） |
| AdamW foreach 门控 | `scripts/train_pytorch.py` | `PLAW_VLA_ADAMW_FOREACH=0`（默认）禁用 foreach 实现，防 +12G 瞬态爆 64G 显存 |
| 权重流式加载 | `scripts/train_pytorch.py` | `safe_open` 逐 tensor 加载，替代 8G 全量 CPU dict（宿主 cgroup 32G 硬墙） |
| ckpt 保存窗防护 | `scripts/train_pytorch.py` | flock 串行 + page cache 逐出（posix_fadvise DONTNEED）+ settle |
| 梯度累积 | `src/openpi/training/config.py` + `scripts/train_pytorch.py` | `grad_accum_steps` 保等效 batch 不动 LR（默认 1 = 原行为） |

## 环境（P2 路线）

venv：`/home/tione/notebook/home/arianliu/venvs/plaw-venv`（py3.10 + 系统 das torch 2.4.1 + transformers 5.0.0）。
构建脚本思路见 `/tmp/setup_plaw_venv.sh`（要点：**venv 场景 unset PYTHONHOME/PYTHONPATH**、`--system-site-packages`、pip 用清华镜像、numpy 钉 1.26.4）。

## 用法

> **2026-10-07 卡位规定**：复现工作**只用 0 卡**，1 卡留给其他项目（恢复双卡另行通知）。
> 单卡保持配方等效 batch 靠梯度累积：等效 batch = batch-size × grad-accum-steps × 卡数。

```bash
source dcu/env.sh
bash dcu/preflight.sh                 # 开工前 30 秒体检

# 环境从零搭建（venv + transformers 补丁一次到位）
bash dcu/setup_venv.sh

# 训练（等效 batch 通过 batch-size x grad-accum-steps 保持配方）
INIT_WEIGHT=checkpoints/pi05_base_pytorch \
  bash dcu/train_stage.sh stage3_finetuning_libero \
  --batch-size 64 --grad-accum-steps 4

# 断点续训（checkpoint 每 5000 步自动保存）
bash dcu/train_stage.sh stage3_finetuning_libero --resume

# 评测服务端
bash dcu/serve_policy.sh checkpoints/stage3_finetuning_libero/<exp>/<step>
```

评测客户端（CPU 仿真，另开终端）见 `examples/libero/README.md`；本机 GPU 不能跑仿真渲染，
客户端用 `MUJOCO_GL=egl`（失败退 `osmesa`），不要给客户端进程设 `HIP_VISIBLE_DEVICES`。

## 本机铁律速记

- **卡位**：复现仅用 0 卡（2026-10-07 起），1 卡归其他项目
- 选卡 `HIP_VISIBLE_DEVICES`；hy-smi 只信 VRAM%（HCU% 假 0.0）
- 宿主 cgroup 32G 硬墙：重 CPU 活（转换/norm stats/下载解压）单跑；ckpt 保存走 SAVE_LOCK
- DeepSpeed/ZeRO 在 das 构建必死；`expandable_segments` 不支持；**numpy 必须 1.26.4**（numpy≥2 会让 das torch CUDA init 失败/挂死）
- venv 内不得出现本地 torch/torchvision dist-info（会骗过 transformers 的版本探测）；装 lerobot 必须 `--no-deps`
- HF 走 hf-mirror.com 且 `HF_HUB_ENABLE_HF_TRANSFER=0`；pip 用清华镜像（pypi.org 本机 RTT 2s）
