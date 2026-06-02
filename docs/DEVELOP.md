# OmniProbe Development Guide

OmniProbe evaluates the dense features of visual foundation models. Everything runs through one entrypoint:

```bash
python -m omniprobe.run task=<task> backbone=<backbone> [task.mode=<mode>]
omniprobe --list-tasks      # tasks and their modes
omniprobe --list-backbones  # available backbone configs
```

Where things live:

| Concern | Code | Config |
|---------|------|--------|
| Backbones | `omniprobe/models/<name>.py` | `configs/backbone/<name>.yaml` |
| Tasks | `omniprobe/tasks/__init__.py` (registry) + `omniprobe/scripts/<script>.py` | `configs/task/<task>.yaml` (+ `configs/<script>.yaml`) |

`omniprobe/run.py` validates the task/mode, builds a runtime context, and dispatches to the task. Backbone capabilities are described by *contracts* (`omniprobe/models/contracts.py`), which the runtime checks before a task loads a model.

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

**2. Contract** — register the model's capabilities in `_BACKBONE_CONTRACTS` (`omniprobe/models/contracts.py`), keyed by the `_target_` string. The positional fields are `(target, supported_outputs, default_global_output, supports_multilayer, supports_layer_selection, input_normalization)`:

```python
"omniprobe.models.your_model.YourModel": BackboneContract(
    "omniprobe.models.your_model.YourModel",
    ("dense", "gap", "cls"),   # supported_outputs
    "gap",                     # default global output (None -> first of cls/gap/map)
    True,                      # supports_multilayer
    True,                      # supports_layer_selection
    "imagenet",                # input normalization ("imagenet" or "raw")
),
```

(Alternatively, the same fields can be set inline in the YAML — `supported_outputs`, `default_global_output`, … — and `get_backbone_contract` will read them — but the registry is the norm.)

**3. Config** — `configs/backbone/your_model.yaml`:

```yaml
_target_: omniprobe.models.your_model.YourModel
output: dense
layer: -1
```

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
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path


def run_task(cfg: DictConfig):
    device = cfg.device
    model = instantiate(cfg.backbone, output="dense").to(device).eval()
    loader = build_loader(cfg.dataset, "test", batch_size=4)
    # ... iterate over loader, run model(images), build a metrics dict ...
    entry = build_result_entry("your_task", "default", model, output_dir, cfg, metrics)
    append_jsonl(resolve_results_path(cfg, "your_task.jsonl"), entry)
```

**2. Script config** — `configs/eval_your_task.yaml` (whatever the script reads: `dataset`, hyperparameters, …).

**3. Task config** — `configs/task/your_task.yaml`:

```yaml
name: your_task
mode: default
dataset:
  path: ${oc.env:YOUR_TASK_ROOT,data/your_dataset}
```

**4. Registry** — add an entry to `SCRIPT_TASKS` in `omniprobe/tasks/__init__.py`. For a single-mode task, the `_simple_dispatch` helper is enough:

```python
"your_task": {
    "modes": ("default",),
    "dispatch": _simple_dispatch("eval_your_task", "omniprobe.scripts.eval_your_task"),
},
```

A dispatch function has signature `dispatch(cfg, mode_name, contract) -> (config_name, module_name, extra_overrides)` and should assert the features it needs, e.g. `contract.require_output("dense", "your_task", mode_name)`. For multi-mode tasks, write a custom dispatch that routes `mode_name` to different `config_name`/`module_name` (see `_correspondence_soco_dispatch`). `validate_task_mode` rejects unknown task names and modes up front.

`run_script_task` (`omniprobe/tasks/script_task.py`) loads `configs/<config_name>.yaml`, merges in the `task.*` and backbone overrides, then imports the module and calls its `run_task(cfg)`.

**Native tasks** (no script delegation) implement `run(cfg, context)` in their module and are registered in `_NATIVE_TASK_MODULES` (e.g. `classification_imagenet_knn`).

---

## Datasets

Dataset classes live in `omniprobe/datasets/` and are loaded by scripts via `build_loader(cfg.dataset, split, batch_size, num_workers)` (`omniprobe/datasets/builder.py`). Roots default to `data/<dataset>` and can be overridden with the dataset's env var (e.g. `SOCO_ROOT`).

## Testing

```bash
pytest tests/ -q        # install the `dev` extra first
```

New backbone configs need no extra test: the contract/config tests glob `configs/backbone/*.yaml` and verify each resolves and instantiates (for hub/fake-loadable backbones).
