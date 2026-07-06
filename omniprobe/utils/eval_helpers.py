from typing import Any, Sequence

import torch
from loguru import logger

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
PERCEPTION_MEAN = [0.5, 0.5, 0.5]
PERCEPTION_STD = [0.5, 0.5, 0.5]
RAW_MEAN = [0.0, 0.0, 0.0]
RAW_STD = [1.0, 1.0, 1.0]


def resolve_mean_std(value) -> tuple[list[float], list[float]]:
    if isinstance(value, str):
        key = value.lower()
        if key == "imagenet":
            return list(IMAGENET_MEAN), list(IMAGENET_STD)
        if key == "clip":
            return list(CLIP_MEAN), list(CLIP_STD)
        if key in ("perception", "halves"):
            return list(PERCEPTION_MEAN), list(PERCEPTION_STD)
        if key in ("raw", "zeros", "none"):
            return list(RAW_MEAN), list(RAW_STD)
        raise ValueError(f"Unsupported normalization preset '{value}'")
    if isinstance(value, dict):
        mean = value.get("mean", [0.0, 0.0, 0.0])
        std = value.get("std", [1.0, 1.0, 1.0])
        return list(mean), list(std)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return list(value), [1.0, 1.0, 1.0]
    raise ValueError(f"Cannot resolve mean/std from {value}")


def global_pool(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return features.mean(dim=(-2, -1))
    return features


def _to_square_patch_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return None
        first = int(value[0])
        if len(value) > 1 and int(value[1]) != first:
            raise ValueError(f"Non-square patch size is not supported: {value}.")
        return first
    return int(value)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _feature_grid_hw(model, image_size: int) -> tuple[int, int]:
    images = torch.randn(1, 3, image_size, image_size, device=_model_device(model))
    feats = model(images)
    if isinstance(feats, list):
        if len(feats) == 0:
            raise ValueError("Model returned an empty feature list.")
        feats = feats[0]
    if feats.ndim != 4:
        raise ValueError(f"Expected 4D dense features, got shape {tuple(feats.shape)}")
    return int(feats.shape[-2]), int(feats.shape[-1])


def _infer_patch_size_from_forward(model, probe_image_size: int) -> int:
    feat_h, feat_w = _feature_grid_hw(model, probe_image_size)
    if feat_h != feat_w:
        raise ValueError(
            f"Expected square feature map for square input, got {(feat_h, feat_w)}"
        )
    if feat_h <= 0:
        raise ValueError(f"Invalid feature map size: {(feat_h, feat_w)}")
    patch_size = int(round(probe_image_size / feat_h))
    if patch_size <= 0:
        raise ValueError(
            f"Failed to infer patch size from input {probe_image_size} and grid {feat_h}"
        )
    return patch_size


def _resolve_patch_size(model, probe_image_size: int) -> tuple[int, str]:
    patch_size = _to_square_patch_size(getattr(model, "patch_size", None))
    if patch_size is not None:
        if patch_size <= 0:
            raise ValueError(f"model.patch_size must be > 0, got {patch_size}")
        return patch_size, "model.patch_size"
    return _infer_patch_size_from_forward(model, probe_image_size), "inferred_from_forward"


def _nearest_multiple(value: int, multiple: int) -> int:
    lower = (value // multiple) * multiple
    upper = lower + multiple
    if lower <= 0:
        return upper
    return lower if value - lower <= upper - value else upper


def resolve_correspondence_image_size(cfg, model) -> dict[str, Any]:
    """Resolve the actual square image size used by dense correspondence tasks."""

    requested_image_size = int(cfg.image_size)
    if requested_image_size <= 0:
        raise ValueError(f"image_size must be > 0, got {requested_image_size}")

    patch_size, patch_size_source = _resolve_patch_size(model, requested_image_size)

    fixed_patched_size = bool(cfg.get("fixed_patched_size", False))
    num_patches = int(cfg.get("num_patches", 60))
    policy = str(cfg.get("image_size_policy", "nearest_multiple"))

    if fixed_patched_size:
        if num_patches <= 0:
            raise ValueError(f"num_patches must be > 0, got {num_patches}")
        effective_image_size = num_patches * patch_size
        verified_grid_hw = _feature_grid_hw(model, effective_image_size)
        if verified_grid_hw != (num_patches, num_patches):
            raise ValueError(
                "fixed_patched_size=True requested "
                f"{num_patches}x{num_patches} patches, but got {verified_grid_hw} "
                f"for image_size={effective_image_size} and patch_size={patch_size} "
                f"({patch_size_source})."
            )
        applied_policy = "fixed_patch_grid"
    else:
        if policy != "nearest_multiple":
            raise ValueError(
                "Unsupported image_size_policy "
                f"'{policy}'. Supported policy: nearest_multiple."
            )
        effective_image_size = _nearest_multiple(requested_image_size, patch_size)
        verified_grid_hw = None
        applied_policy = policy

    grid_size = effective_image_size // patch_size
    return {
        "requested_image_size": requested_image_size,
        "effective_image_size": effective_image_size,
        "image_size_policy": applied_policy,
        "patch_size": patch_size,
        "patch_size_source": patch_size_source,
        "num_patches": num_patches,
        "fixed_patched_size": fixed_patched_size,
        "expected_grid_hw": (grid_size, grid_size),
        "verified_grid_hw": verified_grid_hw,
    }


def log_correspondence_image_size(info: dict[str, Any]) -> None:
    logger.info(
        "Image resolution: requested={} effective={} "
        "(policy={}, fixed_patched_size={}, num_patches={}, "
        "resolved_patch_size={}, patch_size_source={}, expected_grid_hw={}, "
        "verified_grid_hw={})",
        info["requested_image_size"],
        info["effective_image_size"],
        info["image_size_policy"],
        info["fixed_patched_size"],
        info["num_patches"],
        info["patch_size"],
        info["patch_size_source"],
        info["expected_grid_hw"],
        info["verified_grid_hw"],
    )


def correspondence_image_size_result_fields(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_image_size": info["requested_image_size"],
        "effective_image_size": info["effective_image_size"],
        "image_size_policy": info["image_size_policy"],
        "expected_grid_hw": info["expected_grid_hw"],
        "verified_grid_hw": info["verified_grid_hw"],
    }
