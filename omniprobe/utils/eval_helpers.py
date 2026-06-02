
from typing import Sequence, Tuple

import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
PERCEPTION_MEAN = [0.5, 0.5, 0.5]
PERCEPTION_STD = [0.5, 0.5, 0.5]
RAW_MEAN = [0.0, 0.0, 0.0]
RAW_STD = [1.0, 1.0, 1.0]


def resolve_mean_std(value) -> Tuple[list[float], list[float]]:
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
