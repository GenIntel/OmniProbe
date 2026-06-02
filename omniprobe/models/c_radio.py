"""
Backbone definition for NVIDIA C-RADIOv4 via TorchHub.

Based on the RADIO TorchHub model API and aligned with evals/models/radio.py.
"""

import warnings

import torch
import torch.nn.functional as F

from .utils import center_padding, default_multilayers, tokens_to_output


class CRADIOv4Backbone(torch.nn.Module):
    """
    Backbone wrapper for C-RADIOv4 models (TorchHub).

    Args:
        version (str): Version string for TorchHub (e.g., "c-radio_v4-h").
        output (str): One of ["dense", "gap", "cls"].
        return_multilayer (bool): If True, return intermediate layer features.
        force_reload (bool): Force TorchHub to reload the RADIO repository code.
    """

    def __init__(
        self,
        version: str = "c-radio_v4-h",
        output: str = "dense",
        return_multilayer: bool = False,
        image_mean: str = "raw",
        force_reload: bool = False,
    ) -> None:
        super().__init__()

        self.version = version
        self.checkpoint_name = version
        self.image_mean = image_mean

        self.radio = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=self.version,
            progress=True,
            skip_validation=True,
            force_reload=force_reload,
            trust_repo=True,
        )
        self.radio_preprocessor = self.radio.make_preprocessor_external()
        self.radio = self.radio.eval().to(torch.float32)

        assert output in ["dense", "gap", "cls"]
        self.output = output

        patch_gen = self.radio.model.patch_generator
        self.patch_size = patch_gen.patch_size

        feat_dim = self.radio.model.embed_dim

        num_layers = len(self.radio.model.blocks)
        multilayers = default_multilayers(num_layers)

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1]
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def _resize_to_supported(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.radio, "get_nearest_supported_resolution"):
            h, w = images.shape[-2:]
            new_h, new_w = self.radio.get_nearest_supported_resolution(h, w)
            if (new_h, new_w) != (h, w):
                images = F.interpolate(
                    images, (new_h, new_w), mode="bilinear", align_corners=False
                )
        images = center_padding(images, self.patch_size)
        return images

    def _warn_on_range(self, images: torch.Tensor) -> None:
        if not torch.is_floating_point(images):
            return
        with torch.no_grad():
            min_val = float(images.min().item())
            max_val = float(images.max().item())
        if min_val < -0.01 or max_val > 1.01:
            warnings.warn(
                "C-RADIO expects raw inputs in [0, 1] before RADIO conditioning. "
                f"Got min={min_val:.3f}, max={max_val:.3f}.",
                RuntimeWarning,
            )

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = self.radio.model.patch_generator(x)

        for i, blk in enumerate(self.radio.model.blocks):
            x = blk(x)
            if i in self.multilayers:
                features.append(self.radio.model.norm(x))

        return features

    def forward(self, images: torch.Tensor):
        if images.dim() != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {images.shape}")

        images = images.to(dtype=torch.float32)
        self._warn_on_range(images)
        images = self._resize_to_supported(images)
        images = self.radio_preprocessor(images)

        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        intermediate_features = self.forward_features(images)

        outputs = []
        for features in intermediate_features:
            summary = features[:, 0]
            patches = features[:, self.radio.model.patch_generator.num_skip :]
            output = tokens_to_output(self.output, patches, summary, (h, w))
            outputs.append(output)

        return outputs[0] if len(outputs) == 1 else outputs


class CRADIOv3Backbone(CRADIOv4Backbone):
    """
    Backbone wrapper for C-RADIOv3 models (TorchHub).

    Defaults to the base model variant (C-RADIOv3-B).
    """

    def __init__(
        self,
        version: str = "c-radio_v3-b",
        output: str = "dense",
        return_multilayer: bool = False,
        image_mean: str = "raw",
        force_reload: bool = False,
    ) -> None:
        super().__init__(
            version=version,
            output=output,
            return_multilayer=return_multilayer,
            image_mean=image_mean,
            force_reload=force_reload,
        )
