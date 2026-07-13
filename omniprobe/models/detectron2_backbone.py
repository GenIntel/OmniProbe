"""Generic detectron2 backbone adapter for OmniProbe backbones.

Wraps any backbone following the OmniProbe backbone protocol (instantiated
with ``output="dense"`` and ``return_multilayer=True``) together with a
feature-pyramid probe (e.g. ``DPT_FPN``) as a ``detectron2`` Backbone that
emits ``{p2, p3, p4, p5}`` feature maps for detection heads.

This module requires detectron2 and is only imported by the
``detection3d_omni3d`` task.
"""

import torch
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone


class OmniProbeD2Backbone(Backbone):
    """Adapts an OmniProbe backbone + pyramid probe to the detectron2 FPN contract.

    Feature maps come out of the backbone at its patch stride and are mapped
    to nominal strides 4/8/16/32 by the probe. Patch-14 backbones are treated
    as stride 16 at the p4 level, matching the reference Cube R-CNN
    SSL-backbone experiments.
    """

    def __init__(self, model, probe, freeze: bool = True):
        super().__init__()
        self.model = model
        self.probe = probe
        self._freeze = bool(freeze)
        self._out_features = ["p2", "p3", "p4", "p5"]
        self._out_feature_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
        self._out_feature_channels = {
            name: probe.output_dim for name in self._out_features
        }
        if self._freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze:
            # keep normalization/dropout layers of the frozen backbone in eval mode
            self.model.eval()
        return self

    def forward(self, images: torch.Tensor):
        if self._freeze:
            with torch.no_grad():
                feats = self.model(images)
        else:
            feats = self.model(images)

        if not isinstance(feats, (list, tuple)):
            raise TypeError(
                "Backbone returned a single tensor despite return_multilayer; "
                "the detection probe needs 4 feature maps."
            )
        if len(feats) != 4:
            raise ValueError(
                f"Expected 4 multilayer feature maps, got {len(feats)}."
            )

        # pad to even spatial dims so the x2 pyramid additions line up
        feats = [
            f
            if f.shape[-1] % 2 == 0 and f.shape[-2] % 2 == 0
            else F.pad(f, (0, f.shape[-1] % 2, 0, f.shape[-2] % 2))
            for f in feats
        ]
        return self.probe(list(feats))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


def build_rcnn3d_model(d2_cfg, backbone, priors, device):
    """Constructs a Cube R-CNN (RCNN3D) model around an already-built backbone.

    Bypasses the detectron2 registries: the backbone comes from OmniProbe's
    Hydra config, everything else from the detectron2 config node.
    """
    from detectron2.modeling.proposal_generator import build_proposal_generator

    # importing registers RPNWithIgnore / ROIHeads3D in the d2 registries
    from omniprobe.models.vendor.cubercnn.modeling.meta_arch.rcnn3d import RCNN3D
    from omniprobe.models.vendor.cubercnn.modeling.proposal_generator import rpn  # noqa: F401
    from omniprobe.models.vendor.cubercnn.modeling.roi_heads import build_roi_heads

    model = RCNN3D(
        backbone=backbone,
        proposal_generator=build_proposal_generator(d2_cfg, backbone.output_shape()),
        roi_heads=build_roi_heads(d2_cfg, backbone.output_shape(), priors=priors),
        input_format=d2_cfg.INPUT.FORMAT,
        vis_period=0,
        pixel_mean=d2_cfg.MODEL.PIXEL_MEAN,
        pixel_std=d2_cfg.MODEL.PIXEL_STD,
    )
    return model.to(torch.device(device))
