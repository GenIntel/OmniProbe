
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import torch
import torch.nn as nn

from .utils import (
    center_padding,
    default_multilayers as _default_multilayers,
    resolve_pretrained_path,
)

# VGGT uses absolute `from vggt.*` imports internally; make the vendored package importable.
_VGGT_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "vggt"
if str(_VGGT_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_VENDOR_ROOT))

from vggt.models.vggt import VGGT as _VGGT  # noqa: E402


class VGGTBackbone(nn.Module):
    """
    Wrapper around Meta's VGGT for dense correspondence features.
    Supports extracting features from:
    - the full VGGT aggregator (`feature_source="aggregator"`)
    - the fine-tuned DINOv2 patch embed inside the aggregator
      (`feature_source="patch_embed"`).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        return_multilayer: bool = False,
        output: str = "dense",
        layer: int = -1,
        feature_source: str = "aggregator",
        image_mean: Optional[str] = None,
    ) -> None:
        super().__init__()
        if output != "dense":
            raise ValueError("VGGT backbone currently supports only dense output.")
        if feature_source not in {"aggregator", "patch_embed"}:
            raise ValueError(
                "VGGT backbone feature_source must be one of "
                "{'aggregator', 'patch_embed'}."
            )
        self.feature_source = feature_source
        self.image_mean = image_mean

        self.model = _VGGT()

        if checkpoint_path is None:
            default_ckpt = resolve_pretrained_path("vggt", "vggt_1B_commercial.pt")
            checkpoint_path = str(default_ckpt)

        checkpoint_ref = str(checkpoint_path)
        parsed = urlparse(checkpoint_ref)
        is_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)

        if is_url:
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_ref)
            checkpoint_name = Path(parsed.path).stem
            checkpoint_label = checkpoint_ref
        else:
            ckpt_path = Path(checkpoint_ref).expanduser()
            ckpt_path, checkpoint_label, checkpoint_name = self._resolve_checkpoint_path(
                ckpt_path
            )
            if ckpt_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(ckpt_path))
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
            if "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"VGGT checkpoint {checkpoint_label} missing keys: {missing}"
            )
        if unexpected:
            raise RuntimeError(
                f"VGGT checkpoint {checkpoint_label} has unexpected keys: {unexpected}"
            )

        self.model.eval().to(torch.float32)

        self.output = output
        self.return_multilayer = return_multilayer
        if self.feature_source == "aggregator":
            self.patch_size = self.model.aggregator.patch_size
            embed_dim = self.model.aggregator.frame_blocks[0].attn.proj.out_features
            depth = self.model.aggregator.depth
            feat_dim = embed_dim * 2  # concatenated frame/global features
        else:
            patch_size = getattr(self.model.aggregator.patch_embed, "patch_size", 14)
            if isinstance(patch_size, (tuple, list)):
                patch_size = patch_size[0]
            self.patch_size = int(patch_size)
            embed_dim = getattr(self.model.aggregator.patch_embed, "embed_dim", None)
            if embed_dim is None:
                raise RuntimeError(
                    "Could not infer embed_dim from VGGT patch_embed module."
                )
            if not hasattr(self.model.aggregator.patch_embed, "blocks"):
                raise RuntimeError(
                    "VGGT patch_embed module does not expose transformer blocks."
                )
            depth = len(self.model.aggregator.patch_embed.blocks)
            feat_dim = embed_dim

        if return_multilayer:
            multilayers = _default_multilayers(depth)
            self.multilayers = sorted(set(multilayers))
            self.feat_dim = [feat_dim] * len(self.multilayers)
        else:
            layer = depth - 1 if layer == -1 else layer
            if layer < 0 or layer >= depth:
                raise ValueError(
                    f"Requested layer {layer} is out of range for depth={depth}."
                )
            self.multilayers = [layer]
            self.feat_dim = feat_dim

        self.layer = "-".join(str(idx) for idx in self.multilayers)
        self.checkpoint_name = checkpoint_name

    def _resolve_checkpoint_path(
        self, checkpoint_path: Path
    ) -> tuple[Path, str, str]:
        checkpoint_label = str(checkpoint_path)
        checkpoint_name = checkpoint_path.stem
        if checkpoint_path.is_dir():
            safetensors_path = checkpoint_path / "model.safetensors"
            if safetensors_path.is_file():
                return safetensors_path, checkpoint_label, checkpoint_path.name
            raise FileNotFoundError(
                f"VGGT checkpoint directory {checkpoint_path} does not contain "
                "model.safetensors."
            )
        if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"VGGT checkpoint was not found at {checkpoint_path}. "
                    "Provide checkpoint_path pointing to a local file, a Hugging Face "
                    "snapshot directory, or an HTTPS URL."
                )
        return checkpoint_path, checkpoint_label, checkpoint_name

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = center_padding(images, self.patch_size)
        h_tokens = images.shape[-2] // self.patch_size
        w_tokens = images.shape[-1] // self.patch_size

        outputs: List[torch.Tensor] = []
        if self.feature_source == "aggregator":
            batch = images.unsqueeze(1)  # [B, 1, 3, H, W]
            tokens_list, patch_start_idx = self.model.aggregator(batch)

            if not tokens_list:
                raise RuntimeError("VGGT aggregator returned no intermediate tokens.")

            target_layers = set(self.multilayers)
            for idx, tokens in enumerate(tokens_list):
                if idx not in target_layers:
                    continue

                selected = tokens[:, :, patch_start_idx:, :]  # [B, S, P, 2C]
                selected = selected.reshape(selected.shape[0], -1, selected.shape[-1])
                feat = selected.transpose(1, 2).reshape(
                    selected.shape[0], -1, h_tokens, w_tokens
                )
                outputs.append(feat)
                if len(outputs) == len(target_layers):
                    break
        else:
            mean = self.model.aggregator._resnet_mean[:, 0].to(
                device=images.device, dtype=images.dtype
            )
            std = self.model.aggregator._resnet_std[:, 0].to(
                device=images.device, dtype=images.dtype
            )
            images = (images - mean) / std

            patch_embed = self.model.aggregator.patch_embed
            if self.return_multilayer:
                outputs = list(
                    patch_embed.get_intermediate_layers(
                        images,
                        n=self.multilayers,
                        reshape=True,
                        return_class_token=False,
                        norm=True,
                    )
                )
            else:
                tokens = patch_embed(images)
                if isinstance(tokens, dict):
                    tokens = tokens["x_norm_patchtokens"]
                if tokens.ndim != 3:
                    raise RuntimeError(
                        f"Unexpected patch token shape from VGGT patch_embed: {tokens.shape}"
                    )
                feat = tokens.transpose(1, 2).reshape(
                    tokens.shape[0], -1, h_tokens, w_tokens
                )
                outputs.append(feat)

        if not outputs:
            raise RuntimeError("VGGT backbone did not capture any intermediate layers.")

        return outputs if self.return_multilayer else outputs[0]
