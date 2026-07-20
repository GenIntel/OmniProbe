<p align="center">
  <img src="assets/omniprobe.jpg" alt="OmniProbe" width="640">
</p>

**A unified framework for evaluating visual features across dense tasks**

[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE) [![Results](https://img.shields.io/badge/results-live-orange.svg)](https://genintel.github.io/OmniProbe/)

OmniProbe gives 25+ families of visual foundation models a *single* command-line and Python interface for probing their features on correspondence, depth, surface-normal, segmentation, pose, tracking, and classification tasks.

📊 **[Browse the results table](https://genintel.github.io/OmniProbe/)** — benchmark results across models and tasks.

- [Highlights](#highlights)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [What's available](#whats-available)
  - [Tasks](#tasks)
  - [Backbones](#backbones)
- [Usage](#usage)
  - [Command line](#command-line)
  - [3D detection (Omni3D)](#3d-detection-omni3d)
  - [Python API](#python-api)
  - [Configs](#configs)
- [Datasets \& paths](#datasets--paths)
- [Contributing](#contributing)
- [Citation](#citation)
- [License \& acknowledgments](#license--acknowledgments)


## Highlights

- **One CLI for every task** — `python -m omniprobe.run task=<task> backbone=<backbone>` has the same shape whether you are matching keypoints or training a depth probe.
- **87 backbone configs** spanning 25+ model families, all behind one feature interface (`dense` / `cls` / `gap` outputs).
- **7 task families** — correspondence (SPair, SOCO, NAVI, ScanNet, AP-10K), depth, surface normals, segmentation (ADE20K), 3D object pose (ImageNet3D), tracking (TAP-Vid), and kNN / linear classification (ImageNet).
- **Configurable via [Hydra](https://hydra.cc/)** — override any setting from the CLI, or compose your own config layers.
- **CLI *or* Python** — run from the shell or call `omniprobe.evaluate(...)` directly.


## Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/

# 2. Create the environment with core dependencies (Python 3.12, PyTorch cu121)
uv sync

# 3. (Optional) configure cache paths for your machine
cp .env.example .env   # then edit HF_HOME / TORCH_HOME / CUDA_HOME
```

Some backbones need extra dependencies — install only what you use:

| Extra | Enables |
|-------|---------|
| `clip` | CLIP / OpenCLIP / ConvNeXt backbones (`open-clip-torch`) |
| `sam` | SAM backbone (`segment-anything`) |
| `diffusion` | DIFT / Stable Diffusion backbone (`diffusers`) |
| `xformers` | memory-efficient attention |
| `knn` | faiss for ImageNet kNN eval |
| `detection3d` | Omni3D 3D detection task (also needs detectron2 + pytorch3d, see below) |
| `data-processing` | dataset preprocessing helpers |
| `dev` | pytest + pre-commit |
| `all` | `clip,sam,diffusion,xformers,data-processing` |

```bash
uv sync --extra clip          # one extra
uv sync --extra all           # everything above
```

The 3D detection task additionally requires [detectron2](https://github.com/facebookresearch/detectron2) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d), which have no PyPI wheels for recent PyTorch and must be built from source against your installed torch/CUDA (they are therefore not part of any extra):

```bash
pip install "git+https://github.com/facebookresearch/detectron2.git" --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip install -e ".[detection3d]"
```

Build requirements: `CUDA_HOME` must point to a CUDA toolkit whose major version matches your torch build (e.g. CUDA 12.x for `torch+cu12x`), with a host compiler nvcc accepts (GCC ≤ 13 for CUDA 12). On a machine without a GPU, set `TORCH_CUDA_ARCH_LIST` (e.g. `"8.0;9.0"` for A100/H100) so the extensions are compiled for the GPUs you will run on.

<details>
<summary>pip fallback</summary>

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[all,knn,dev]"
```
</details>

The code for backbones that build on external repositories (CroCo, I-JEPA, Perception, VGGT, MetaCLIP, PIXIO) is **vendored** under `omniprobe/models/vendor/` — there are no git submodules to fetch. Those models only need their checkpoint files downloaded (see [docs/MODELS.md](./docs/MODELS.md)); DINO/DINOv2, C-RADIO, DUNE and V-JEPA 2 are pulled from `torch.hub` on first use. DINOv3 is pulled from Hugging Face Hub (gated — request access to the relevant `facebook/dinov3-*` repos and set `HF_TOKEN` or run `huggingface-cli login`).


## Quickstart

Every evaluation runs through one entrypoint:

```bash
python -m omniprobe.run task=<task_config> backbone=<backbone>
```

A minimal SOCO correspondence run with a hub-loaded DINOv2 backbone (no checkpoint files needed; you only need the SOCO dataset configured — see [Datasets & paths](#datasets--paths)):

```bash
python -m omniprobe.run task=correspondence_soco backbone=dinov2_b14
```

The same call from Python:

```python
import omniprobe

result = omniprobe.evaluate(
    task="correspondence_soco",
    backbone="dinov2_b14",
)
print(result)
```


## What's available

The CLI is the live source of truth — these always reflect the installed configs:

```bash
omniprobe --list-tasks        # or: python -m omniprobe.run --list-tasks
omniprobe --list-backbones
```

### Tasks

| Family | Datasets |
|--------|----------|
| Correspondence | SPair-71k, SOCO, NAVI, ScanNet, AP-10K |
| Depth | NYU, NAVI |
| Surface normals | NYU, NAVI |
| Segmentation | ADE20K |
| Pose | ImageNet3D |
| 3D detection | Omni3D (ARKitScenes default; indoor/outdoor/full presets) |
| Tracking | TAP-Vid DAVIS |
| Classification | ImageNet |

### Backbones

87 configs across the families below. Pass any config name as `backbone=<name>`; see **[docs/MODELS.md](./docs/MODELS.md)** for the full per-config table (weight source and supported output modes).

| Family | Example configs | Weights |
|--------|-----------------|---------|
| DINO / DINOv2 | `dino_b16`, `dinov2_b14`, `dinov2_l14`, `dinov2_b14_reg` | torch.hub |
| DINOv3 | `dinov3_vitb16`, `dinov3_vitl16`, `dinov3_vitl16_sat` | HF Hub (`transformers`, gated) |
| C-RADIO | `c_radio_3_b`, `c_radio_4_h` | torch.hub |
| DUNE | `dune_vitb14`, `dune_vits14_448` | torch.hub |
| V-JEPA 2 | `vjepa2_1_base`, `vjepa2_1_large` | torch.hub / ckpt |
| CLIP / OpenCLIP | `clip_b16`, `clip_l14`, `openclip_vitl14_laion2b` | open_clip / ckpt |
| ConvNeXt | `clip_convnext`, `convnext_in22k` | open_clip / timm |
| DeiT-III | `deit3_b16`, `deit3_l16` | timm |
| iBOT | `ibot_b16`, `ibot_l16_in22k` | local ckpt |
| MAE | `mae_b16`, `mae_l16`, `mae_h14` | HF Hub |
| SigLIP | `siglip_b16`, `siglip_l16` | timm |
| SAM | `sam_base`, `sam_large`, `sam_huge` | local ckpt |
| MetaCLIP 2 | `metaclip2_vitb16`, `metaclip2_vitl14` | vendored + ckpt |
| PIXIO | `pixio_vitb16`, `pixio_vitl16` | vendored + ckpt |
| Perception | `perception_b16_512`, `perception_l14_448` | vendored + ckpt |
| CroCo | `crocov2` | vendored + ckpt |
| I-JEPA | `ijepa_vith16_448` | vendored + ckpt |
| VGGT | `vggt`, `vggt_dino` | vendored + ckpt |
| DIY-SC | `dinov2_b14_diy_sc` | torch.hub |
| MiDaS | `midas_l16` | torch.hub |
| DIFT / Stable Diffusion | `dift_sd21`, `dift_sd15` | HF Hub (`diffusion` extra) |
| LVLM visual encoders | `qwen2_5_vl_7b`, `internvl3_5_8b`, `llava_ov_7b` | HF Hub (`transformers`) |


## Usage

### Command line

```bash
# First configure dataset roots and (optionally) caches — see "Datasets & paths".

# Correspondence
python -m omniprobe.run task=correspondence_spair backbone=dino_b16
python -m omniprobe.run task=correspondence_soco backbone=dinov2_b14 task.soft_eval=true
python -m omniprobe.run task=correspondence_soco backbone=dinov2_b14 \
  task.pair_subdir=PairAnnotations/cross
python -m omniprobe.run task=correspondence_navi backbone=dino_b16
python -m omniprobe.run task=tracking_tapvid backbone=dinov2_b14

# Dense probes / segmentation
python -m omniprobe.run task=depth backbone=dino_b16
python -m omniprobe.run task=snorm backbone=dino_b16
python -m omniprobe.run task=segmentation_ade20k backbone=dinov2_b14

# ImageNet classification
python -m omniprobe.run task=classification_imagenet_knn    backbone=dinov2_b14 task.data_root=/path/to/imagenet
python -m omniprobe.run task=classification_imagenet_linear backbone=dinov2_b14 task.data_root=/path/to/imagenet
```

### 3D detection (Omni3D)

The 3D detection task trains Cube R-CNN heads on top of a frozen backbone and reports AP2D/AP3D. It needs two things the other tasks don't:

1. **Extra dependencies** — detectron2 and PyTorch3D built from source plus the `detection3d` extra; see the install commands in [Installation](#installation).
2. **The Omni3D data** — annotations and ARKitScenes images under `OMNI3D_ROOT`; see [data_processing/README.md](./data_processing/README.md#omni3d-3d-detection) for the download and layout.

Training and evaluation on ARKitScenes (the default):

```bash
# full training run (ARKitScenes, frozen backbone, 116k iterations)
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14

# multi-GPU: single node, N processes; solver.ims_per_batch is the TOTAL
# batch size across GPUs (default: num_gpus=4, ims_per_batch=32)
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  task.system.num_gpus=4

# evaluate a trained checkpoint (no training); visualize_predictions
# additionally renders 3D cuboid overlays + BEV for every 50th test image
# into <run_dir>/inference/iter_final/<dataset>/vis/
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  task.eval_only=true task.weights=/path/to/model_final.pth \
  task.visualize_predictions=true

# resume an interrupted run: reuse its output directory
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  task.resume=true hydra.run.dir=outputs/<date>/<time>_detection3d_omni3d_dinov2_b14
```

**Other Omni3D datasets.** Preset tasks cover the standard splits — `detection3d_omni3d_in` (SUN RGB-D + Hypersim + ARKitScenes, 38 categories), `detection3d_omni3d_out` (nuScenes + KITTI, 11), and `detection3d_omni3d_full` (all six, 50); run them exactly like the default task once the corresponding datasets are downloaded. For custom combinations, override the dataset lists and let the category set resolve from a preset name (`omni3d`, `omni3d_in`, `omni3d_out`, or any split name):

```bash
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  'task.datasets.train=[KITTI_train,KITTI_val]' 'task.datasets.test=[KITTI_test]' \
  task.datasets.category_names=KITTI_test task.datasets.num_classes=null
```

When evaluating a checkpoint, keep `category_names`/`num_classes` matching the training run — the detection heads are sized for those categories. The default solver follows the frozen-backbone ARKitScenes recipe (AdamW 1e-3, batch 32, 116k iterations); the upstream Cube R-CNN recipe used larger batches for the bigger splits (128 indoor, 192 full), so expect to tune batch/lr/iterations there.

### Python API

```python
omniprobe.evaluate(
    task,                 # task config name, e.g. "correspondence_spair"
    backbone,             # backbone config name, e.g. "dinov2_b14"
    device="auto",        # "cuda" | "cpu" | "auto"
    **task_overrides,     # forwarded onto cfg.task, e.g. data_root=... or soft_eval=True
)
```

```python
import omniprobe

print(omniprobe.available_backbones())   # all backbone config names
print(omniprobe.available_tasks())       # all task names

result = omniprobe.evaluate(
    task="correspondence_soco_linear_probe",
    backbone="dinov2_b14",
    data_root="/path/to/SOCOv1",
)
```

### Configs

Configuration lives in two main Hydra layers:

1. **Runtime** — [`configs/run.yaml`](./configs/run.yaml): default `task`, `backbone`, and runtime-only settings such as `device`.
2. **Backbone configs** — [`configs/backbone/`](./configs/backbone): model constructor settings plus the explicit input normalization preset (`image_mean`, e.g. `imagenet`, `clip`, `perception`, or `raw`).
3. **Task configs** — [`configs/task/`](./configs/task): explicit public defaults for `python -m omniprobe.run`. Distinct protocols get distinct task config names, e.g. `correspondence_soco.yaml` and `correspondence_soco_linear_probe.yaml`. Script-backed tasks also declare their backend module in a small `runner:` block.

Precedence for script-backed tasks is:

```text
selected task config < CLI/Python task overrides < runner overrides
```

Practical rule:

- Change `configs/task/<task_config>.yaml` for user-facing task protocol defaults.
- Change `configs/backbone/<backbone>.yaml` for model-specific input normalization. Runtime plumbing forwards `${backbone.image_mean}` to datasets and legacy scripts.
- Correspondence tasks use `task.image_size` as the requested protocol size. SOCO, SPair, and AP-10K resolve it to the nearest multiple of the backbone patch size before resizing images/keypoints; logs and result rows include both `requested_image_size` and `effective_image_size`.
- Per-run logs and artifacts are written to `outputs/<date>/<run>/`. Aggregate JSONL summaries stay in `results/`.
- Each task config writes to its own `results/<task>.jsonl` (e.g. `correspondence_soco.jsonl` vs `correspondence_soco_linear_probe.jsonl`), so distinct protocols are separated by file rather than by a record field. In-protocol toggles such as `task.soft_eval=true` stay recoverable from the per-record embedded `config`.
- Use Hydra CLI overrides for one-off runs:

```bash
# One-off dataset root override
python -m omniprobe.run task=correspondence_spair backbone=dinov2_b14 \
  task.data_root=/path/to/SPair-71k

# Change linear-probe training settings
python -m omniprobe.run task=correspondence_soco_linear_probe backbone=dinov2_b14 \
  task.train.epochs=20 task.train.lr=0.0005
```

Under the hood, most tasks delegate to evaluation scripts in `omniprobe/scripts/` (via their `run_task(cfg)` function); `classification_imagenet_knn` and `classification_imagenet_linear` are implemented natively in `omniprobe/tasks/`. The runtime entrypoint is [`omniprobe/run.py`](./omniprobe/run.py) and the task registry lives in [`omniprobe/tasks/__init__.py`](./omniprobe/tasks/__init__.py).


## Datasets & paths

Each task reads its dataset root from an environment variable, defaulting to `data/<dataset>` — the layout produced by the download guide. Following **[data_processing/README.md](./data_processing/README.md)** therefore works out of the box from the repo root; override the variable to point elsewhere:

```bash
export SOCO_ROOT=/path/to/SOCOv1   # optional; defaults to data/SOCOv1
python -m omniprobe.run task=correspondence_soco backbone=dinov2_b14
```

Downloaded checkpoints default to `checkpoints/` (override with `OMNIPROBE_PRETRAINED_MODELS`); backbone code that builds on external repositories is vendored under `omniprobe/models/vendor/`. See [docs/MODELS.md](./docs/MODELS.md) for the per-backbone checkpoint env vars (`CROCO_CKPT`, `VGGT_CKPT`, …).


## Contributing

Contributions of new backbones, tasks, and datasets are welcome.

- **Add a backbone:** use [`omniprobe/models/siglip.py`](./omniprobe/models/siglip.py) as a template (it implements the `BackboneProtocol` from `omniprobe/models/utils.py`), then add `configs/backbone/<name>.yaml` with a `_target_` pointing at your class, and register its capability contract in `omniprobe/models/contracts.py`.
- **Add a task:** register it in `omniprobe/tasks/__init__.py` and add a `configs/task/<task>.yaml`.
- **Run the tests:** `pytest tests/ -q` (install the `dev` extra first).

See **[docs/DEVELOP.md](./docs/DEVELOP.md)** for a more detailed developer guide, and **[docs/ROADMAP.md](./docs/ROADMAP.md)** for planned features and open ideas.


## Citation

If you use OmniProbe in your research, please consider giving a star ⭐ and cite:

```bibtex
@article{duenkel2026soco,
  title         = {SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation Models},
  author        = {D{\"u}nkel, Olaf and Sunagad, Basavaraj and Wang, Haoran and
                   Hoffmann, David T. and Theobalt, Christian and Kortylewski, Adam},
  journal       = {arXiv preprint arXiv:2605.31597},
  year          = {2026}
}
```


## License & acknowledgments

OmniProbe is released under the [MIT License](./LICENSE).

This project builds on several open-source works; see [docs/THIRD_PARTY_LICENSES.md](./docs/THIRD_PARTY_LICENSES.md) for full attribution and their licenses. We especially thank [Probing the 3D Awareness of Visual Foundation Models](https://arxiv.org/abs/2404.08636) (CVPR 2024), whose implementation this framework heavily builds upon.

Note that some vendored components carry more restrictive licenses than MIT — in particular the Cube R-CNN code used by the 3D detection task (`omniprobe/models/vendor/cubercnn/`) is CC BY-NC 4.0 (**non-commercial**), and MetaCLIP/I-JEPA/PIXIO carry non-commercial terms as well. These subtrees are only imported when you use the corresponding backbones or tasks.
