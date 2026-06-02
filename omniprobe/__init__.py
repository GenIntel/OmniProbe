"""OmniProbe -- A unified framework for evaluating visual features across dense tasks."""

__version__ = "1.0.0"

from omniprobe.tasks import available_tasks


def available_backbones() -> list[str]:
    """Return sorted names of all backbone configs."""
    from pathlib import Path

    config_dir = Path(__file__).resolve().parent.parent / "configs" / "backbone"
    if not config_dir.exists():
        return []
    return sorted(p.stem for p in config_dir.glob("*.yaml"))


def evaluate(
    task: str,
    backbone: str,
    mode: str = "default",
    device: str = "auto",
    **task_overrides,
):
    """Run a task evaluation programmatically (without Hydra CLI).

    Args:
        task: Task name (e.g. "spair", "depth").
        backbone: Backbone config name (e.g. "dinov2_b14").
        mode: Task mode (e.g. "nn", "train", "default").
        device: Device string ("cuda", "cpu", "auto").
        **task_overrides: Additional task-level config overrides.

    Returns:
        The result dict produced by the task's run function.
    """
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    from omniprobe.runtime import build_runtime_context
    from omniprobe.tasks import load_task_module

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="run",
            overrides=[
                f"task={task}",
                f"backbone={backbone}",
                f"task.mode={mode}",
                f"device={device}",
            ],
        )

    if task_overrides:
        with open_dict(cfg):
            for key, value in task_overrides.items():
                cfg.task[key] = value

    context = build_runtime_context(cfg)
    task_module = load_task_module(str(cfg.task.name))
    return task_module.run(cfg, context)
