from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger

from .utils import (
    center_padding,
    default_multilayers,
    resolve_pretrained_reference,
    tokens_to_output,
)
from .vendor.pixio.pixio import (
    pixio_vitb16,
    pixio_vitl16,
    pixio_vith16,
    pixio_vit1b16,
    pixio_vit5b16,
)


class PIXIO(nn.Module):
    """
    Wrapper around the PIXIO ViT model for use in omniprobe evaluations.
    Follows the same interface as other backbone models (DINO, MAE, etc.)
    """

    VARIANTS: Dict[str, Dict[str, Union[int, None]]] = {
        "vitb16": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 16},
        "vitl16": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "patch_size": 16},
        "vith16": {"embed_dim": 1280, "depth": 32, "num_heads": 16, "patch_size": 16},
        "vit1b16": {"embed_dim": 1536, "depth": 48, "num_heads": 24, "patch_size": 16},
        "vit5b16": {"embed_dim": 3072, "depth": 48, "num_heads": 32, "patch_size": 16},
    }

    def __init__(
        self,
        arch: str = "vitb16",
        *,
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
        weights: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        
        if arch not in self.VARIANTS:
            raise ValueError(f"Unsupported PIXIO architecture '{arch}'. Available: {list(self.VARIANTS.keys())}")
        
        variant = self.VARIANTS[arch]
        if output not in {"cls", "gap", "dense"}:
            raise ValueError(f"Unsupported output type '{output}'.")

        model_factories = {
            "vitb16": pixio_vitb16,
            "vitl16": pixio_vitl16,
            "vith16": pixio_vith16,
            "vit1b16": pixio_vit1b16,
            "vit5b16": pixio_vit5b16,
        }
        
        # Create the model (optionally with pretrained weights)
        weights_path = resolve_pretrained_reference(weights) if weights is not None else None
        pretrained_arg = str(weights_path) if weights_path is not None and weights_path.exists() else None
        self.model = model_factories[arch](pretrained=pretrained_arg)
        
        if pretrained_arg:
            logger.info(f"Loaded PIXIO weights from {pretrained_arg}")
        
        self.model = self.model.eval().to(torch.float32)
        
        self.output = output
        self.arch = arch
        self.patch_size = variant["patch_size"]
        self.n_cls_tokens = 8  # Fixed in pixio models

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
                raise ValueError(f"Requested layer {target_layer} outside valid range [0, {num_layers - 1}]")
            self.multilayers = [target_layer]
            self.feat_dim = self.base_feat_dim

        weight_tag = (
            Path(str(weights)).stem if weights is not None and Path(str(weights)).exists() else "random"
        )
        self.checkpoint_name = f"pixio_{arch}_{weight_tag}"
        self.layer = "-".join(str(idx) for idx in self.multilayers)

    def forward(self, images: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Pad images to ensure divisibility by patch_size
        images = center_padding(images, self.patch_size)
        
        h, w = images.shape[-2:]
        feat_h = h // self.patch_size
        feat_w = w // self.patch_size
        
        # Get features from specified layers
        features = self.model(images, block_ids=self.multilayers)
        
        outputs: List[torch.Tensor] = []
        for feat_dict in features:
            # Use normalized patch tokens (after LayerNorm)
            dense_tokens = feat_dict['patch_tokens_norm']
            # Average the multiple class tokens to get a single cls token
            cls_token = feat_dict['cls_tokens_norm'].mean(dim=1)
            
            feat = tokens_to_output(self.output, dense_tokens, cls_token, (feat_h, feat_w))
            outputs.append(feat)

        return outputs[0] if len(outputs) == 1 else outputs
