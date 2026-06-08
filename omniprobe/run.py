import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from omniprobe.runtime import (
    build_runtime_context,
    configure_run_logging,
)
from omniprobe.tasks import load_task_module, validate_task


def _print_tasks():
    from omniprobe.tasks import SCRIPT_TASKS, _NATIVE_TASK_MODULES

    print("Available tasks:\n")
    for name in sorted(SCRIPT_TASKS):
        print(f"  {name:<38s} {SCRIPT_TASKS[name]['runner'].module}")
    for name in sorted(_NATIVE_TASK_MODULES):
        print(f"  {name:<38s} (native)")
    print()


def _print_backbones():
    from omniprobe import available_backbones

    print("Available backbones:\n")
    for name in available_backbones():
        print(f"  {name}")
    print()


@hydra.main(config_path="../configs", config_name="run", version_base=None)
def main(cfg: DictConfig):
    context = build_runtime_context(cfg)
    with configure_run_logging(context.output_dir):
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        validate_task(cfg)
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
