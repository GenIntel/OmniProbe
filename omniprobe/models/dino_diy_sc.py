"""
DIY-SC: DINOv2 backbone with a learned AggregationNetwork head for
semantic correspondence.

Loads the head from the official TorchHub repo::

    torch.hub.load('odunkel/DIY-SC-torchhub', 'agg_dino', pretrained=True)

Available hub entrypoints:
    agg_dino, agg_dino_384, agg_dino_128,
    agg_dino_in3d, agg_dino_in3d_spair,
    agg_sd_dino, agg_sd_dino_in3d, agg_sd_dino_in3d_spair
"""

import torch
import torch.nn as nn
from loguru import logger

from .dino import DINO as DINOBackbone


class DINO(nn.Module):
    """DINOv2 backbone + DIY-SC aggregation head."""

    def __init__(
        self,
        dino_name="dinov2",
        model_name="vitb14",
        output="dense",
        layer=-1,
        return_multilayer=False,
        hub_entrypoint="agg_dino",
    ):
        super().__init__()

        self.backbone = DINOBackbone(
            dino_name=dino_name,
            model_name=model_name,
            output=output,
            layer=layer,
            return_multilayer=return_multilayer,
        )

        self.checkpoint_name = f"{self.backbone.checkpoint_name}_diy_sc_{hub_entrypoint}"
        self.patch_size = self.backbone.patch_size
        self.output = self.backbone.output
        self.layer = self.backbone.layer
        self.multilayers = self.backbone.multilayers
        self.feat_dim = self.backbone.feat_dim

        logger.info(f"Loading DIY-SC head from TorchHub: {hub_entrypoint}")
        self.aggre_net = torch.hub.load(
            "odunkel/DIY-SC-torchhub",
            hub_entrypoint,
            pretrained=True,
            trust_repo=True,
        )
        self.aggre_net.eval()

    def forward(self, images):
        outputs = self.backbone(images)
        single = not isinstance(outputs, list)
        if single:
            outputs = [outputs]

        with torch.no_grad():
            outputs[-1] = self.aggre_net(outputs[-1])

        return outputs[0] if single else outputs
