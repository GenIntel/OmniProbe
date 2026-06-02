"""Task registry for the flat runtime.

Script-backed tasks are declared in SCRIPT_TASKS — each entry maps a
task name to its modes and a dispatch function that returns the
(config_name, module_name, extra_overrides) triple.

Native tasks (classification_imagenet_knn, classification_imagenet_linear)
have their own run() and are registered directly in _NATIVE_TASK_MODULES.
"""

from importlib import import_module
from types import SimpleNamespace

from omniprobe.models.contracts import get_backbone_contract, require_supported_mode
from omniprobe.runtime import log_runtime_header
from omniprobe.tasks.script_task import run_script_task


# ---------------------------------------------------------------------------
# Script-backed task dispatch functions
# ---------------------------------------------------------------------------

def _correspondence_spair_dispatch(cfg, mode_name, contract):
    contract.require_output("dense", "correspondence_spair", mode_name)
    if mode_name == "linear_probe":
        return "eval_correspondence_linear_probe_spair", "omniprobe.scripts.eval_correspondence_linear_probe_spair", None
    return "eval_correspondence_spair", "omniprobe.scripts.eval_correspondence_spair", {"soft_eval": mode_name == "soft_argmax"}


def _correspondence_soco_dispatch(cfg, mode_name, contract):
    contract.require_output("dense", "correspondence_soco", mode_name)
    if mode_name == "linear_probe":
        return "eval_correspondence_linear_probe_soco", "omniprobe.scripts.eval_correspondence_linear_probe_soco", None
    return "eval_correspondence_soco", "omniprobe.scripts.eval_correspondence_soco", {
        "soft_eval": mode_name == "soft_argmax",
    }


def _segmentation_ade20k_dispatch(cfg, mode_name, contract):
    contract.require_output("dense", "segmentation_ade20k", mode_name)
    extra = None
    if "data_root" in cfg.task and cfg.task.data_root is not None:
        extra = {"dataset": {"root": str(cfg.task.data_root)}}
    if mode_name == "eval":
        return "eval_segmentation_ade20k", "omniprobe.scripts.eval_segmentation_ade20k", extra
    return "train_segmentation_ade20k", "omniprobe.scripts.train_segmentation_ade20k", extra


def _pose_imagenet3d_dispatch(cfg, mode_name, contract):
    if mode_name == "ep":
        contract.require_output("dense", "pose_imagenet3d", mode_name)
    else:
        contract.resolve_global_output()
    config_name = "train_pose_ep_imagenet3d" if mode_name == "ep" else "train_pose_imagenet3d"
    return config_name, "omniprobe.scripts.train_pose_imagenet3d", None


def _multilayer_probe_dispatch(task_name, config_name, module_name):
    """Create a dispatch function for multilayer probe tasks (depth, snorm)."""
    def _dispatch(cfg, mode_name, contract):
        contract.require_output("dense", task_name, mode_name)
        if not contract.supports_multilayer:
            raise ValueError(f"Task '{task_name}' requires multilayer backbone features.")
        return config_name, module_name, {"backbone": {"return_multilayer": True}}
    return _dispatch


def _simple_dispatch(config, module, required_output="dense", extra_overrides_fn=None):
    """Create a dispatch function for simple tasks with no mode-dependent routing."""
    def _dispatch(cfg, mode_name, contract):
        contract.require_output(required_output, config, mode_name)
        extra = extra_overrides_fn(cfg) if extra_overrides_fn else None
        return config, module, extra
    return _dispatch


def _data_root_override(cfg):
    """Forward task.data_root to the script config."""
    if "data_root" in cfg.task and cfg.task.data_root is not None:
        return {"data_root": str(cfg.task.data_root)}
    return None


SCRIPT_TASKS: dict[str, dict] = {
    "correspondence_spair": {
        "modes": ("nn", "soft_argmax", "linear_probe"),
        "dispatch": _correspondence_spair_dispatch,
    },
    "correspondence_soco": {
        "modes": ("nn", "soft_argmax", "linear_probe"),
        "dispatch": _correspondence_soco_dispatch,
    },
    "segmentation_ade20k": {
        "modes": ("train", "eval"),
        "dispatch": _segmentation_ade20k_dispatch,
    },
    "pose_imagenet3d": {
        "modes": ("default", "ep"),
        "dispatch": _pose_imagenet3d_dispatch,
    },
    "depth": {
        "modes": ("default",),
        "dispatch": _multilayer_probe_dispatch("depth", "train_depth", "omniprobe.scripts.train_depth"),
    },
    "snorm": {
        "modes": ("default",),
        "dispatch": _multilayer_probe_dispatch("snorm", "train_snorm", "omniprobe.scripts.train_snorm"),
    },
    "correspondence_navi": {
        "modes": ("default",),
        "dispatch": _simple_dispatch("eval_correspondence_navi", "omniprobe.scripts.eval_correspondence_navi"),
    },
    "tracking_tapvid": {
        "modes": ("default",),
        "dispatch": _simple_dispatch("eval_tracking_tapvid", "omniprobe.scripts.eval_tracking_tapvid"),
    },
    "correspondence_scannet": {
        "modes": ("default",),
        "dispatch": _simple_dispatch(
            "eval_correspondence_scannet", "omniprobe.scripts.eval_correspondence_scannet",
            extra_overrides_fn=_data_root_override,
        ),
    },
    "correspondence_geometric_soco": {
        "modes": ("default",),
        "dispatch": _simple_dispatch("eval_correspondence_geometric_soco", "omniprobe.scripts.eval_correspondence_geometric_soco"),
    },
    "correspondence_ap10k": {
        "modes": ("default",),
        "dispatch": _simple_dispatch(
            "eval_correspondence_ap10k", "omniprobe.scripts.eval_correspondence_ap10k",
            extra_overrides_fn=_data_root_override,
        ),
    },
}

# Native tasks (implemented directly, no script delegation)
_NATIVE_TASK_MODULES = {
    "classification_imagenet_knn": "omniprobe.tasks.imagenet_knn",
    "classification_imagenet_linear": "omniprobe.tasks.imagenet_linear",
}


# ---------------------------------------------------------------------------
# Generic runner for script-backed tasks
# ---------------------------------------------------------------------------

def _run_script_task(task_name: str, cfg, context):
    entry = SCRIPT_TASKS[task_name]
    mode_name = str(cfg.task.mode)
    require_supported_mode(mode_name, entry["modes"], task_name)
    log_runtime_header(task_name, mode_name, context)
    contract = get_backbone_contract(cfg.backbone)
    config_name, module_name, extra_overrides = entry["dispatch"](cfg, mode_name, contract)
    return run_script_task(cfg, task_name, config_name, module_name, extra_overrides=extra_overrides)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_task_mode(cfg) -> None:
    """Validate task name and mode early, before backbone loading."""
    task_name = str(cfg.task.name)
    if task_name not in SCRIPT_TASKS and task_name not in _NATIVE_TASK_MODULES:
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {available_tasks()}"
        )
    if task_name in SCRIPT_TASKS:
        mode = str(cfg.task.mode)
        supported = SCRIPT_TASKS[task_name]["modes"]
        if mode not in supported:
            raise ValueError(
                f"Unsupported mode '{mode}' for task '{task_name}'. "
                f"Supported modes: {list(supported)}"
            )


def available_tasks() -> list[str]:
    return sorted([*SCRIPT_TASKS, *_NATIVE_TASK_MODULES])


def load_task_module(task_name: str):
    """Load and return a task module with a run(cfg, context) function.

    For native tasks, returns the module directly.
    For script-backed tasks, returns a lightweight wrapper with a run() function.
    """
    if task_name in _NATIVE_TASK_MODULES:
        return import_module(_NATIVE_TASK_MODULES[task_name])

    if task_name in SCRIPT_TASKS:
        wrapper = SimpleNamespace()
        wrapper.run = lambda cfg, context, _name=task_name: _run_script_task(_name, cfg, context)
        return wrapper

    raise ValueError(
        f"Unknown task '{task_name}'. Available tasks: {available_tasks()}"
    )
