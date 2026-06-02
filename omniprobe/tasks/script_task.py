from importlib import import_module
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from omniprobe.runtime import resolve_device


_CONTROL_TASK_KEYS = {"name", "mode"}
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_TASK_CONFIG_DIR = _CONFIG_DIR / "task"


def load_script_config(config_name: str):
    if GlobalHydra.instance().is_initialized():
        return hydra.compose(config_name=config_name)
    with hydra.initialize_config_dir(
        config_dir=str(_CONFIG_DIR),
        version_base=None,
    ):
        return hydra.compose(config_name=config_name)


def _load_task_defaults(task_name: str):
    task_cfg_path = _TASK_CONFIG_DIR / f"{task_name}.yaml"
    if task_cfg_path.exists():
        return OmegaConf.load(task_cfg_path)
    return OmegaConf.create({})


def _build_task_override_payload(cfg, task_name: str) -> dict:
    task_payload = OmegaConf.to_container(cfg.task, resolve=True)
    default_payload = OmegaConf.to_container(_load_task_defaults(task_name), resolve=False)
    overrides = {}
    for key, value in task_payload.items():
        if key in _CONTROL_TASK_KEYS:
            continue
        if key in default_payload and default_payload[key] == value:
            continue
        overrides[key] = value
    return overrides


def _split_top_level_overrides(overrides: dict, script_cfg):
    known = {key: value for key, value in overrides.items() if key in script_cfg}
    extra = {key: value for key, value in overrides.items() if key not in script_cfg}
    return known, extra


def build_script_cfg(
    cfg,
    task_name: str,
    config_name: str,
    extra_overrides: dict | None = None,
):
    script_cfg = load_script_config(config_name)
    merged = [script_cfg]
    direct_overrides = {}

    task_overrides = _build_task_override_payload(cfg, task_name)
    if task_overrides:
        task_known, task_extra = _split_top_level_overrides(task_overrides, script_cfg)
        if task_known:
            merged.append(OmegaConf.create(task_known))
        direct_overrides.update(task_extra)

    if "device" in cfg:
        resolved_device = str(resolve_device(str(cfg.device)))
        if "device" in script_cfg:
            merged.append(OmegaConf.create({"device": resolved_device}))
        else:
            direct_overrides["device"] = resolved_device

    backbone_overrides = OmegaConf.create({})
    if extra_overrides is not None:
        non_backbone_overrides = {
            key: value for key, value in extra_overrides.items() if key != "backbone"
        }
        if non_backbone_overrides:
            extra_known, extra_direct = _split_top_level_overrides(
                non_backbone_overrides,
                script_cfg,
            )
            if extra_known:
                merged.append(OmegaConf.create(extra_known))
            direct_overrides.update(extra_direct)
        if "backbone" in extra_overrides:
            backbone_overrides = OmegaConf.create(extra_overrides["backbone"])

    merged_cfg = OmegaConf.merge(*merged)

    merged_backbone = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(cfg.backbone, resolve=True)),
        backbone_overrides,
    )

    with open_dict(merged_cfg):
        merged_cfg.backbone = merged_backbone
        for key, value in direct_overrides.items():
            merged_cfg[key] = value
    return merged_cfg


def run_script_task(
    cfg,
    task_name: str,
    config_name: str,
    module_name: str,
    extra_overrides: dict | None = None,
):
    script_cfg = build_script_cfg(
        cfg,
        task_name,
        config_name,
        extra_overrides=extra_overrides,
    )
    module = import_module(module_name)
    if not hasattr(module, "run_task"):
        raise AttributeError(
            f"Task script module '{module_name}' does not define run_task(cfg)."
        )
    return module.run_task(script_cfg)
