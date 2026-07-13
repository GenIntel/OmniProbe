# OmniProbe Development Guide

OmniProbe evaluates the dense features of visual foundation models. Everything runs through one entrypoint:

```bash
python -m omniprobe.run task=<task_config> backbone=<backbone>
omniprobe --list-tasks      # runnable task configs
omniprobe --list-backbones  # available backbone configs
```

Where things live:

| Concern | Code | Config |
|---------|------|--------|
| Backbones | `omniprobe/models/<name>.py` | `configs/backbone/<name>.yaml` |
| Tasks | `omniprobe/tasks/__init__.py` (registry) + `omniprobe/scripts/<script>.py` | `configs/task/<task>.yaml` |

`omniprobe/run.py` validates the selected task config, builds a runtime context, and dispatches to the task. Backbone capabilities are described by *contracts* (`omniprobe/models/contracts.py`), which the runtime checks before a task loads a model.

---

## Backbone interface

Every backbone is an `nn.Module` satisfying `BackboneProtocol` (`omniprobe/models/utils.py`):

```python
class BackboneProtocol(Protocol):
    checkpoint_name: str
    patch_size: int
    layer: str
    output: str                      # one of: "dense", "cls", "gap" (+ "map" for SigLIP)
    feat_dim: int | list[int]

    def forward(self, images: Tensor) -> Tensor | list[Tensor]: ...
```

Conventions:

- `__init__` accepts `output="dense"`, `layer=-1`, `return_multilayer=False`.
- `feat_dim` is an `int` (and `forward` returns one `Tensor`) when `return_multilayer=False`; it is a `list[int]` (and `forward` returns a `list[Tensor]`) when `True`.
- Pick layer indices with `default_multilayers(num_layers)`; pad inputs with `center_padding(images, patch_size)`; build the requested output with `tokens_to_output(output, dense_tokens, cls_token, (h, w))` (handles `dense` / `cls` / `gap`).

---

## Add a backbone

Use `omniprobe/models/siglip.py` as the reference template.

**1. Wrapper** — `omniprobe/models/your_model.py`:

```python
import torch
from .utils import center_padding, default_multilayers, tokens_to_output


class YourModel(torch.nn.Module):
    def __init__(self, output="dense", layer=-1, return_multilayer=False):
        super().__init__()
        assert output in ["dense", "gap", "cls"]
        self.output = output
        self.checkpoint_name = "your_model"

        self.vit = ...                       # load pretrained weights here
        self.patch_size = ...

        feat_dim = self.vit.embed_dim
        multilayers = default_multilayers(len(self.vit.blocks))
        if return_multilayer:
            self.feat_dim = [feat_dim] * len(multilayers)
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            self.multilayers = [multilayers[-1] if layer == -1 else layer]
        self.layer = "-".join(str(x) for x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        # ... run the transformer, collect token maps at self.multilayers ...
        outputs = [
            tokens_to_output(self.output, dense_tokens, cls_token, (h, w))
            for dense_tokens, cls_token in collected
        ]
        return outputs[0] if len(self.multilayers) == 1 else outputs
```

**2. Contract** — register the model's capabilities in `_BACKBONE_CONTRACTS` (`omniprobe/models/contracts.py`), keyed by the `_target_` string. The positional fields are `(target, supported_outputs, default_global_output, supports_multilayer, supports_layer_selection)`:

```python
"omniprobe.models.your_model.YourModel": BackboneContract(
    "omniprobe.models.your_model.YourModel",
    ("dense", "gap", "cls"),   # supported_outputs
    "gap",                     # default global output (None -> first of cls/gap/map)
    True,                      # supports_multilayer
    True,                      # supports_layer_selection
),
```

(Alternatively, the same fields can be set inline in the YAML — `supported_outputs`, `default_global_output`, … — and `get_backbone_contract` will read them — but the registry is the norm.)

**3. Config** — `configs/backbone/your_model.yaml`:

```yaml
_target_: omniprobe.models.your_model.YourModel
image_mean: imagenet
output: dense
layer: -1
```

Use `image_mean: clip`, `perception`, or `raw` when the backbone expects that input convention. Dataset configs and legacy script payloads receive `${backbone.image_mean}`.

Run it: `python -m omniprobe.run task=correspondence_soco backbone=your_model`. The config-glob test in `tests/test_backbone_contracts.py` automatically checks that every `configs/backbone/*.yaml` resolves to a contract.

> The runtime instantiates backbones via `instantiate_backbone_for_output(cfg, output, return_multilayer, device, layer)`, which strips contract metadata keys and passes `output` / `return_multilayer` / `layer`. Eval scripts that don't need contract checks often call `instantiate(cfg.backbone, output="dense", return_multilayer=...)` directly.

---

## Add a task

Most tasks are **script-backed**: a thin registry entry routes to an evaluation / training script. (Two classification tasks are *native* — see the bottom.)

**1. Script** — `omniprobe/scripts/eval_your_task.py` exposing `run_task(cfg)`:

```python
from hydra.utils import instantiate
from omegaconf import DictConfig

from omniprobe.datasets.builder import build_loader
from omniprobe.runtime import (
    append_jsonl,
    artifact_dir,
    build_result_entry,
    resolve_output_dir,
    resolve_results_path,
)


def run_task(cfg: DictConfig):
    device = cfg.device
    output_dir = resolve_output_dir(cfg)
    model = instantiate(cfg.backbone, output="dense").to(device).eval()
    loader = build_loader(cfg.dataset, "test", batch_size=4)
    # ... iterate over loader, run model(images), build a metrics dict ...
    predictions_dir = artifact_dir(cfg, "predictions")
    entry = build_result_entry("your_task", model, output_dir, cfg, metrics)
    append_jsonl(resolve_results_path(cfg, "your_task.jsonl"), entry)
```

**2. Task config** — `configs/task/your_task.yaml`. This is the public protocol layer for `python -m omniprobe.run`; put the deliberate task defaults here. If another protocol has meaningfully different defaults, add an explicit task config such as `configs/task/your_task_linear_probe.yaml` rather than hiding the differences in a shared group.

If a variant only toggles behavior inside the same protocol, prefer an explicit override instead of a new task config. For example, soft-argmax correspondence and SOCO cross-pair evaluation are run through their base task configs:

```bash
python -m omniprobe.run task=correspondence_spair backbone=dinov2_b14 \
  task.soft_eval=true
python -m omniprobe.run task=correspondence_soco backbone=dinov2_b14 \
  task.pair_subdir=PairAnnotations/cross
```

```yaml
name: your_task
dataset:
  path: ${oc.env:YOUR_TASK_ROOT,data/your_dataset}
```

Script-backed task configs also declare the backend module:

```yaml
runner:
  module: omniprobe.scripts.eval_your_task
  required_output: dense
```

Use `required_output: global` for tasks that consume the backbone's default global output. Use `require_multilayer: true` for tasks that need multilayer features — the runtime then forwards `return_multilayer=true` to the backbone automatically. Add an explicit `extra_overrides:` block only for any other backbone/script overrides a task must force.

`run_script_task` (`omniprobe/tasks/script_task.py`) flattens `cfg.task` into the legacy script config, excluding control/internal keys (`name`, `runner`), then applies backbone/device and runner overrides before importing the module and calling `run_task(cfg)`. Precedence is `selected task config < CLI/Python task overrides < runner overrides`.

Correspondence scripts should use `resolve_correspondence_image_size` from `omniprobe.utils.eval_helpers` before constructing datasets. This keeps resized images, keypoints, masks, and dense feature grids in one coordinate frame: fixed image-size protocols round the requested `task.image_size` to the nearest patch-size multiple, while fixed-patch protocols use `num_patches * patch_size`.

**Native tasks** (no script delegation) implement `run(cfg, context)` in their module and are registered in `_NATIVE_TASK_MODULES` (e.g. `classification_imagenet_knn`).

**The detectron2 exception.** `detection3d_omni3d` is script-backed like the rest, but internally drives the vendored Cube R-CNN stack (`omniprobe/models/vendor/cubercnn/`): the Hydra task config is bridged into a detectron2 `CfgNode` (`build_d2_cfg`), datasets are registered in detectron2's `DatasetCatalog` instead of going through `build_loader`, and multi-GPU runs use `detectron2.engine.launch`. The backbone still comes from the standard Hydra config — any dense multilayer backbone works via `OmniProbeD2Backbone` (`omniprobe/models/detectron2_backbone.py`). To resume a run, reuse its directory: `python -m omniprobe.run task=detection3d_omni3d ... task.resume=true hydra.run.dir=<previous run dir>`.

---

## Datasets

Dataset classes live in `omniprobe/datasets/` and are loaded by scripts via `build_loader(cfg.dataset, split, batch_size, num_workers)` (`omniprobe/datasets/builder.py`). Roots default to `data/<dataset>` and can be overridden with the dataset's env var (e.g. `SOCO_ROOT`).

## Testing

```bash
pytest tests/ -q        # install the `dev` extra first
```

New backbone configs need no extra test: the contract/config tests glob `configs/backbone/*.yaml` and verify each resolves and instantiates (for hub/fake-loadable backbones).
