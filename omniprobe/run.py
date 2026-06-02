import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from omniprobe.runtime import build_runtime_context
from omniprobe.tasks import load_task_module, validate_task_mode


def _print_tasks():
    from omniprobe.tasks import SCRIPT_TASKS, _NATIVE_TASK_MODULES

    print("Available tasks:\n")
    for name in sorted([*SCRIPT_TASKS, *_NATIVE_TASK_MODULES]):
        if name in SCRIPT_TASKS:
            modes = ", ".join(SCRIPT_TASKS[name]["modes"])
            print(f"  {name:<30s} modes: {modes}")
        else:
            print(f"  {name:<30s} (native)")
    print()


def _print_backbones():
    from omniprobe import available_backbones

    print("Available backbones:\n")
    for name in available_backbones():
        print(f"  {name}")
    print()


@hydra.main(config_path="../configs", config_name="run", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    validate_task_mode(cfg)
    context = build_runtime_context(cfg)
    task_module = load_task_module(str(cfg.task.name))
    return task_module.run(cfg, context)


def main_cli():
    if "--list-tasks" in sys.argv:
        _print_tasks()
        return
    if "--list-backbones" in sys.argv:
        _print_backbones()
        return
    main()


if __name__ == "__main__":
    main_cli()
