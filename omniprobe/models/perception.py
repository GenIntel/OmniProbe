
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from .utils import (
    center_padding,
    default_multilayers as _default_multilayers,
    resolve_pretrained_path,
    resolve_pretrained_reference,
    tokens_to_output,
)
from .vendor.perception_models.core.vision_encoder import pe


class PerceptionBackbone(nn.Module):
    """
    Wrapper for Meta's Perception models (Perception-1) vision encoders.

    Requires a local clone of the Semantic Correspondence Benchmarking
    repository and offline checkpoints downloaded separately.
    """

    def __init__(
        self,
        model_type: str,
        checkpoint_path: Optional[str] = None,
        layer: int = -1,
        return_multilayer: bool = False,
        output: str = "dense",
    ) -> None:
        super().__init__()
        if output not in ["dense", "gap"]:
            raise ValueError("Perception backbone currently supports only dense or gap output.")

        if checkpoint_path is None:
            default_ckpt = resolve_pretrained_path("pe", f"{model_type}.pt")
            checkpoint_path = str(default_ckpt)
        else:
            checkpoint_path = str(resolve_pretrained_reference(checkpoint_path))

        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Perception checkpoint was not found at {ckpt_path}. "
                "Provide checkpoint_path pointing to the downloaded *.pt file."
            )

        self.model = pe.VisionTransformer.from_config(
            model_type, pretrained=True, checkpoint_path=str(ckpt_path)
        )
        self.model.pool_type = "none"
        self.model.proj_dim = None

        self.model.eval().to(torch.float32)

        self.patch_size = self.model.patch_size
        self.output = output
        self.return_multilayer = return_multilayer
        self.total_layers = self.model.layers
        embed_dim = self.model.transformer.width

        if return_multilayer:
            multilayers = _default_multilayers(self.total_layers)
            self.multilayers = sorted(set(multilayers))
            self.feat_dim = [embed_dim] * len(self.multilayers)
            self.layer = "-".join(str(idx) for idx in self.multilayers)
        else:
            if layer == -1:
                layer = self.total_layers - 1
            if layer < 0 or layer >= self.total_layers:
                raise ValueError(
                    f"Requested layer {layer} outside valid range [0, {self.total_layers - 1}]"
                )
            self.multilayers = [layer]
            self.feat_dim = embed_dim
            self.layer = str(layer)

        self.checkpoint_name = ckpt_path.stem

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = center_padding(images, self.patch_size)
        h_tokens = images.shape[-2] // self.patch_size
        w_tokens = images.shape[-1] // self.patch_size

        batch_size = images.shape[0]
        grid_h, grid_w = h_tokens, w_tokens

        x = self.model.conv1(images)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.model.width)

        if self.model.use_cls_token:
            cls_embed = self.model.class_embedding.view(1, 1, -1).expand(batch_size, -1, -1)
            x = torch.cat([cls_embed, x], dim=1)

        if self.model.use_abs_posemb:
            pos_embed = self.model._sample_abs_posemb(grid_h, grid_w).to(x.dtype)
            x = x + pos_embed

        if self.model.use_rope2d:
            self.model.rope.update_grid(x.device, grid_h, grid_w)

        x = self.model.ln_pre(x)

        outputs: List[torch.Tensor] = []
        target_layers = set(self.multilayers)

        for idx, block in enumerate(self.model.transformer.resblocks):
            x = block(x)
            if idx in target_layers:
                tokens = self.model.ln_post(x)
                cls_token = tokens[:, 0] if self.model.use_cls_token else None
                patch_tokens = tokens[:, 1:, :] if self.model.use_cls_token else tokens

                feat = tokens_to_output(
                    self.output, patch_tokens, cls_token, (grid_h, grid_w)
                )
                outputs.append(feat)
                if len(outputs) == len(target_layers):
                    break

        if not outputs:
            raise RuntimeError("Perception backbone did not capture any intermediate layers.")

        return outputs if self.return_multilayer else outputs[0]
