<h1 align="center">PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies</h1>

<p align="center">
  <!-- TODO: add the arXiv badge once the paper ID is public. -->
  <a href="https://rainyrobo.github.io/PLaW-VLA"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>

<p align="center">
  <strong><em>Conference on Robot Learning (CoRL) 2026</em></strong>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="PLaW-VLA overview and evaluation results" width="100%">
</p>

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone --recurse-submodules https://github.com/RainyRobo/PLaW-VLA.git
cd PLaW-VLA

GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.12
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
bash scripts/install_transformers_patch.sh
```

`uv` installs the required Python version and CUDA-enabled packages. An NVIDIA GPU and a compatible driver are required. Run the last command again after every subsequent `uv sync`.

> **Submodules:** of the five registered submodules, only `third_party/libero` is
> required (for LIBERO evaluation). `third_party/aloha`, `third_party/robotwin`,
> `third_party/libero-plus`, and `third_party/vjepa2` are optional references and
> are not imported by any code path, so you can skip them with
> `git submodule update --init third_party/libero` to save clone time.

## Downloading Assets

Download all required weights and data with one command:

```bash
uv run python scripts/download_assets.py --stage all
```

| Asset | Source | Local path |
| --- | --- | --- |
| π<sub>0.5</sub> base checkpoint, converted to PyTorch | `gs://openpi-assets/checkpoints/pi05_base` | `checkpoints/pi05_base_pytorch/` |
| V-JEPA2 encoder | [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | Hugging Face cache |
| PaliGemma tokenizer | `gs://big_vision/paligemma_tokenizer.model` | `~/.cache/plaw-vla/` |
| LIBERO dataset | [`RainyBot/libero_v3_eef`](https://huggingface.co/datasets/RainyBot/libero_v3_eef) | `data/libero_v3_eef/` |
| Normalization statistics | Computed from the dataset | `assets/<config>/libero_v3_eef/` |

Existing files are skipped. The training scripts run the same step automatically, so this command is optional.

## Training

Run the three stages in order. Stage II continues from the latest Stage I checkpoint, and Stage III from the latest Stage II checkpoint.

```bash
bash scripts/run_stage1_world_model_pretraining.sh
bash scripts/run_stage2_pretraining.sh
bash scripts/run_stage3_finetuning_libero.sh
```

`NUM_GPUS` defaults to the number of visible GPUs and must divide the batch size. Checkpoints are written to `checkpoints/<config>/<config>/<step>/`. Logs go to the Weights & Biases project `plaw-vla`; run `wandb login` before training.

To use another dataset, add a `TrainConfig` in [`src/openpi/training/config.py`](src/openpi/training/config.py), then compute normalization statistics:

```bash
uv run python scripts/compute_norm_stats.py --config-name <config_name>
```

[`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py) is a reference converter. See [docs/norm_stats.md](docs/norm_stats.md).

## Evaluation

The LIBERO client uses Python 3.8 and the `third_party/libero` submodule.

```bash
git submodule update --init --recursive
uv sync --project examples/libero --python 3.8 --frozen
```

Start the policy server. `--policy.dir` is a checkpoint step directory containing `model.safetensors`.

```bash
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=stage3_finetuning_libero \
  --policy.dir=checkpoints/stage3_finetuning_libero/stage3_finetuning_libero/<step>
```

Start the simulator client in another process. Other suites are `libero_object`, `libero_goal`, and `libero_10`.

```bash
uv run --project examples/libero --frozen python examples/libero/main.py \
  --task-suite-name libero_spatial
```

Multi-suite and multi-GPU evaluation: [`examples/libero/README.md`](examples/libero/README.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

<p>
  Please cite this work if you use the code. It appears at the <strong><em>Conference on Robot Learning (CoRL) 2026</em></strong>.
</p>

```bibtex
@inproceedings{liu2026plawvla,
  title={PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies},
  author={Liu, Yu and Guo, Hetian and Huang, Tianlv and Cai, Ziyi and Chen, Wudi and Wang, Hantang and Liu, Qiutong and Peng, Yingzhi and Han, Wei and Tang, Peijun and Wang, Jianan and Fan, Zipei and Zha, Zhiyuan and Song, Xuan},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```

## Acknowledgments

This repository builds on [openpi](https://github.com/Physical-Intelligence/openpi), [V-JEPA 2](https://github.com/facebookresearch/vjepa2), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LeRobot](https://github.com/huggingface/lerobot), and [Big Vision](https://github.com/google-research/big_vision).
