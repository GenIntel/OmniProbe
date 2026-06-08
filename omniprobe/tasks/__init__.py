"""Task registry for the flat runtime."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from omniprobe.models.contracts import get_backbone_contract
from omniprobe.runtime import log_runtime_header
from omniprobe.tasks.script_task import run_script_task


_NATIVE_TASK_MODULES = {
    "classification_imagenet_knn": "omniprobe.tasks.imagenet_knn",
    "classification_imagenet_linear": "omniprobe.tasks.imagenet_linear",
}
_TASK_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "task"


def _load_script_tasks() -> dict[str, dict]:
    tasks = {}
    for path in sorted(_TASK_CONFIG_DIR.glob("*.yaml")):
        cfg = OmegaConf.load(path)
        if "runner" not in cfg:
            continue
        task_name = str(cfg.name)
        if task_name != path.stem:
            raise ValueError(
                f"Task config '{path.name}' declares name='{task_name}'; "
                f"name must match the filename stem '{path.stem}'."
            )
        tasks[task_name] = {"runner": cfg.runner}
    return tasks


SCRIPT_TASKS: dict[str, dict] = _load_script_tasks()


def _runner_extra_overrides(cfg) -> dict:
    runner = cfg.task.runner
    extra = OmegaConf.to_container(
        OmegaConf.create(runner.get("extra_overrides", {})),
        resolve=True,
    )
    if runner.get("require_multilayer", False):
        extra.setdefault("backbone", {})["return_multilayer"] = True
    return extra or None


def _validate_runner_contract(task_name: str, cfg) -> None:
    contract = get_backbone_contract(cfg.backbone)
    runner = cfg.task.runner
    required_output = str(runner.get("required_output", "dense"))
    if required_output == "global":
        contract.resolve_global_output()
    else:
        contract.require_output(required_output, task_name)
    if runner.get("require_multilayer", False) and not contract.supports_multilayer:
        raise ValueError(f"Task '{task_name}' requires multilayer backbone features.")


# ---------------------------------------------------------------------------
# Generic runner for script-backed tasks
# ---------------------------------------------------------------------------

def _run_script_task(task_name: str, cfg, context):
    log_runtime_header(task_name, context)
    _validate_runner_contract(task_name, cfg)
    extra_overrides = _runner_extra_overrides(cfg)
    return run_script_task(
        cfg,
        str(cfg.task.runner.module),
        extra_overrides=extra_overrides,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_task(cfg) -> None:
    """Validate the task name early, before backbone loading."""
    task_name = str(cfg.task.name)
    if task_name not in SCRIPT_TASKS and task_name not in _NATIVE_TASK_MODULES:
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {available_tasks()}"
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
