from importlib import import_module
from omegaconf import OmegaConf, open_dict

from omniprobe.models.contracts import prepare_backbone_instantiate_cfg
from omniprobe.runtime import resolve_device, resolve_output_dir


_CONTROL_TASK_KEYS = {"name", "runner"}
_ROOT_PAYLOAD_KEYS = ("dataset", "optimizer", "probe")


def build_script_cfg(
    cfg,
    extra_overrides: dict | None = None,
):
    task_payload = OmegaConf.to_container(cfg.task, resolve=True)
    script_cfg = OmegaConf.create(
        {
            key: value
            for key, value in task_payload.items()
            if key not in _CONTROL_TASK_KEYS
        }
    )

    with open_dict(script_cfg):
        script_cfg.task_name = str(cfg.task.name)
        script_cfg.image_mean = str(cfg.backbone.image_mean)
        script_cfg.output_dir = str(resolve_output_dir())

    if "device" in cfg:
        resolved_device = str(resolve_device(str(cfg.device)))
        with open_dict(script_cfg):
            script_cfg.device = resolved_device

    with open_dict(script_cfg):
        for key in _ROOT_PAYLOAD_KEYS:
            if key in cfg and key not in script_cfg:
                script_cfg[key] = OmegaConf.create(
                    OmegaConf.to_container(cfg[key], resolve=True)
                )

    backbone_overrides = OmegaConf.create({})
    if extra_overrides is not None:
        direct_overrides = {
            key: value for key, value in extra_overrides.items() if key != "backbone"
        }
        if direct_overrides:
            script_cfg = OmegaConf.merge(script_cfg, OmegaConf.create(direct_overrides))
        if "backbone" in extra_overrides:
            backbone_overrides = OmegaConf.create(extra_overrides["backbone"])

    merged_backbone = prepare_backbone_instantiate_cfg(
        OmegaConf.merge(
            OmegaConf.create(OmegaConf.to_container(cfg.backbone, resolve=True)),
            backbone_overrides,
        )
    )

    with open_dict(script_cfg):
        script_cfg.backbone = merged_backbone
    return script_cfg


def run_script_task(
    cfg,
    module_name: str,
    extra_overrides: dict | None = None,
):
    script_cfg = build_script_cfg(
        cfg,
        extra_overrides=extra_overrides,
    )
    module = import_module(module_name)
    if not hasattr(module, "run_task"):
        raise AttributeError(
            f"Task script module '{module_name}' does not define run_task(cfg)."
        )
    return module.run_task(script_cfg)
