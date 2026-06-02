import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf


class RuntimeContext:
    def __init__(self, cfg, device: torch.device, output_dir: Path) -> None:
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name is None or device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def build_runtime_context(cfg) -> RuntimeContext:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = str(cfg.device) if "device" in cfg else "auto"
    return RuntimeContext(cfg, resolve_device(device_name), output_dir)


def resolve_results_path(cfg, default_filename: str) -> Path:
    """Resolve the results log path from config, with a per-task default filename."""
    base = str(cfg.results_dir) if "results_dir" in cfg else "results"
    p = Path(base)
    if p.suffix == ".jsonl":
        return p
    return p / default_filename


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def config_to_string(cfg) -> str:
    return str(OmegaConf.to_container(cfg, resolve=True))


def build_result_entry(
    task_name: str,
    mode_name: str,
    model,
    output_dir,
    cfg,
    metrics: dict,
    **extra,
) -> dict:
    entry = {
        "time": datetime.now().strftime("%d%m%Y-%H%M"),
        "task": task_name,
        "mode": mode_name,
        "backbone": getattr(model, "checkpoint_name", None),
        "patch_size": getattr(model, "patch_size", None),
        "layer": str(getattr(model, "layer", "")),
        "output": getattr(model, "output", None),
        "metrics": metrics,
        "output_dir": str(output_dir),
        "config": config_to_string(cfg),
    }
    entry.update(extra)
    return entry


def flatten_features(feats):
    if isinstance(feats, (list, tuple)):
        return torch.cat(list(feats), dim=1)
    return feats


def extract_backbone_features(model, images, normalize: bool = False, pool: str | None = None):
    feats = model(images)
    feats = flatten_features(feats)
    if pool == "global" and feats.ndim == 4:
        feats = feats.mean(dim=(-2, -1))
    if normalize:
        feats = F.normalize(feats, p=2, dim=1)
    return feats


def resolve_image_mean(backbone_contract, cfg_image_mean):
    if cfg_image_mean is not None:
        return cfg_image_mean
    if backbone_contract.input_normalization:
        return backbone_contract.input_normalization
    return "imagenet"


def log_runtime_header(task_name: str, mode_name: str, context: RuntimeContext) -> None:
    logger.info(
        f"Running task='{task_name}' mode='{mode_name}' on device='{context.device}' "
        f"output_dir='{context.output_dir}'"
    )
