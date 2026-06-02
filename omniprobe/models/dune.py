import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger

from .utils import center_padding, default_multilayers, tokens_to_output


class DUNE(nn.Module):
    """
    Wrapper around the DUNE ViT model for use in omniprobe evaluations.
    DUNE models are loaded via torch.hub from naver/dune.
    Follows the same interface as other backbone models (DINO, MAE, etc.)
    """

    VARIANTS: Dict[str, Dict[str, Union[str, int]]] = {
        # ViT-Small variants
        "vits14_448": {
            "hub_name": "dune_vitsmall_14_448_encoder",
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "patch_size": 14,
        },
        # ViT-Base variants
        "vitb14_336": {
            "hub_name": "dune_vitbase_14_336_encoder",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "patch_size": 14,
        },
        "vitb14_448": {
            "hub_name": "dune_vitbase_14_448_encoder",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "patch_size": 14,
        },
        "vitb14_448_paper": {
            "hub_name": "dune_vitbase_14_448_paper_encoder",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "patch_size": 14,
        },
    }

    def __init__(
        self,
        arch: str = "vitb14_448_paper",
        *,
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
    ) -> None:
        super().__init__()

        if arch not in self.VARIANTS:
            raise ValueError(
                f"Unsupported DUNE architecture '{arch}'. Available: {list(self.VARIANTS.keys())}"
            )

        variant = self.VARIANTS[arch]
        if output not in {"cls", "gap", "dense"}:
            raise ValueError(f"Unsupported output type '{output}'.")

        # Load model from torch hub
        hub_name = variant["hub_name"]
        self.model = torch.hub.load("naver/dune", hub_name, trust_repo=True)
        self.model = self.model.eval().to(torch.float32)

        logger.info(f"Loaded DUNE model: {hub_name}")

        self.output = output
        self.arch = arch
        self.patch_size = variant["patch_size"]

        feat_dim = int(variant["embed_dim"])
        self.base_feat_dim = feat_dim

        num_layers = variant["depth"]
        multilayers = sorted(set(default_multilayers(num_layers)))

        if return_multilayer:
            self.multilayers = multilayers
            self.feat_dim = [self.base_feat_dim] * len(self.multilayers)
        else:
            target_layer = multilayers[-1] if layer == -1 else layer
            if not (0 <= target_layer < num_layers):
                raise ValueError(
                    f"Requested layer {target_layer} outside valid range [0, {num_layers - 1}]"
                )
            self.multilayers = [target_layer]
            self.feat_dim = self.base_feat_dim

        self.checkpoint_name = f"dune_{arch}"
        self.layer = "-".join(str(idx) for idx in self.multilayers)

    def forward(self, images: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Pad images to ensure divisibility by patch_size
        images = center_padding(images, self.patch_size)

        h, w = images.shape[-2:]
        feat_h = h // self.patch_size
        feat_w = w // self.patch_size

        # Use DINOv2-style get_intermediate_layers API
        # Returns tuple of (patch_tokens, cls_token) for each layer if return_class_token=True
        outputs_list = self.model.get_intermediate_layers(
            images,
            n=self.multilayers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        outputs: List[torch.Tensor] = []
        for patch_tokens, cls_token in outputs_list:
            feat = tokens_to_output(self.output, patch_tokens, cls_token, (feat_h, feat_w))
            outputs.append(feat)

        return outputs[0] if len(outputs) == 1 else outputs
