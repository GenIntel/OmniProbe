# Third-Party Licenses

This file documents the third-party code used in this project, including
source repositories, copyright holders, licenses, and modifications made.

Vendored packages under `omniprobe/models/vendor/` retain their original LICENSE
files. For custom or non-standard licenses, the full text is in the
referenced file rather than reproduced here.

---

## Non-Vendored Code

### Probe3D

- **Source**: https://github.com/mbanani/probe3d
- **Copyright**: 2024 Mohamed El Banani
- **License**: MIT (see [full text below](#mit-license))
- **Used for**: Core probing framework, depth/surface-normal probes, SPair-71k
  correspondence evaluation, dataset loaders, evaluation infrastructure
- **Modifications**: Restructured, extended with additional tasks and backbones

### NeCo

- **Source**: https://github.com/mbanani/neco
- **Copyright**: Mohamed El Banani
- **License**: MIT (see [full text below](#mit-license))
- **Used for**: ADE20K segmentation linear finetuning recipe and transforms
- **Modifications**: Adapted for the unified evaluation runtime

### FoMo Point Tracking

- **Source**: https://github.com/gorkaydemir/fomo_point_tracking
- **Copyright**: Gorkay Aydemir et al.
- **License**: No explicit license in repository
- **Used for**: TAP-Vid point tracking evaluation and metrics
- **Files**: `omniprobe/utils/tapvid.py` (adapted)
- **Modifications**: Refactored metric computation into standalone utilities

### ImageNet3D

- **Source**: https://github.com/wufeim/imagenet3d
- **Copyright**: Wufei Ma et al.
- **License**: No explicit license in repository
- **Used for**: 3D pose estimation evaluation
- **Modifications**: Adapted for the unified evaluation runtime

### DINO / DINOv2

- **Source**: https://github.com/facebookresearch/dino / https://github.com/facebookresearch/dinov2
- **Copyright**: Facebook, Inc. and its affiliates
- **License**: Apache 2.0 (see [full text below](#apache-license-20))
- **Used for**: kNN evaluation recipe
- **Modifications**: Adapted for the unified evaluation runtime

### DeiT

- **Source**: https://github.com/facebookresearch/deit
- **Copyright**: 2020-present Facebook, Inc. and its affiliates
- **License**: Apache 2.0 (see [full text below](#apache-license-20))
- **Used for**: ViT architecture utilities
- **Files**: `omniprobe/models/deit_utils.py` (copied and modified)
- **Modifications**: Extracted model builder functions; removed unused code

---

## Vendored Code (`omniprobe/models/vendor/`)

### CroCo

- **Source**: https://github.com/naver/croco
- **Commit**: `5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb`
- **Copyright**: Naver Corporation
- **License**: CC BY-NC-SA 4.0
- **Files**: `models/blocks.py`, `models/croco.py`, `models/croco_downstream.py`,
  `models/masking.py`, `models/pos_embed.py`, `models/curope/`
- **Modifications**: Relative imports (`from models.X` → `from .X`)

### I-JEPA

- **Source**: https://github.com/facebookresearch/ijepa
- **Commit**: `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- **Copyright**: Facebook Research
- **License**: CC BY-NC 4.0
- **Files**: `src/models/vision_transformer.py`, `src/masks/utils.py`, `src/utils/tensors.py`
- **Modifications**: Relative imports (`from src.X` → `from ..X`)

### VGGT

- **Source**: https://github.com/facebookresearch/vggt
- **Commit**: `8492456ce358ee9a4fe3274e36d73106b640fb5c`
- **Copyright**: Meta Platforms, Inc.
- **License**: VGGT License (see `omniprobe/models/vendor/vggt/LICENSE.txt`)
- **Files**: `vggt/models/`, `vggt/layers/`, `vggt/heads/`, `vggt/utils/`, `vggt/dependency/`
- **Modifications**: None

### Perception Models (PE)

- **Source**: https://github.com/facebookresearch/perception_models
- **Commit**: `c4d4af6537c51d4fafe16eecb4097bc05621de5d`
- **Copyright**: Meta Platforms, Inc.
- **License**: Apache 2.0 (see `omniprobe/models/vendor/perception_models/LICENSE.PE`)
- **Files**: `core/vision_encoder/__init__.py`, `config.py`, `pe.py`, `rope.py`
- **Modifications**: Relative imports (`from core.vision_encoder.X` → `from .X`)

### Pixio

- **Source**: https://github.com/facebookresearch/pixio
- **Commit**: `b83555d6581f4906f6f1fc1ae0abb04f622db32f`
- **Copyright**: Meta Platforms, Inc.
- **License**: FAIR Noncommercial Research License v1 (see `omniprobe/models/vendor/pixio/LICENSE`)
- **Files**: `pixio/__init__.py`, `pixio/pixio.py`, `pixio/layers/`
- **Modifications**: Relative imports (`from layers.X` → `from .layers.X`)

### MetaCLIP

- **Source**: https://github.com/facebookresearch/MetaCLIP
- **Commit**: `f47f7841f6a91cc5676729a3d125519393d87d1e`
- **Copyright**: Facebook Research
- **License**: CC BY-NC 4.0 (see `omniprobe/models/vendor/metaclip/LICENSE`)
- **Files**: `src/mini_clip/` (11 files), `src/training/checkpoint.py`
- **Modifications**: Relative imports (`from src.mini_clip.X` → `from .X`)

### Omni3D / Cube R-CNN

- **Source**: https://github.com/facebookresearch/omni3d
- **Copyright**: Meta Platforms, Inc. and affiliates
- **License**: CC BY-NC 4.0 (see `omniprobe/models/vendor/cubercnn/LICENSE`,
  which also covers the ARKitScenes and Objectron dataset licenses)
- **Files**: `cubercnn/data/`, `cubercnn/evaluation/`, `cubercnn/modeling/`
  (meta_arch, proposal_generator, roi_heads), `cubercnn/solver/`,
  `cubercnn/util/`, `cubercnn/vis/`, `cubercnn/config.py`
- **Used for**: The `detection3d_omni3d` task (Cube R-CNN detection heads,
  Omni3D data pipeline, AP3D evaluation, and prediction visualization)
- **Modifications**: Import paths rewritten to the vendored package;
  training-time visualization, classic CNN backbones, and model-zoo modules
  removed; registry-based model construction replaced by direct construction
  with an OmniProbe backbone adapter; backbone-specific config keys removed
- **Note**: This vendored subtree is licensed for **non-commercial use only**.
  It is imported exclusively by the `detection3d_omni3d` task; the rest of
  OmniProbe is unaffected.
