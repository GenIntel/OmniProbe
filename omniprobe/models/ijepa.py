
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    center_padding,
    default_multilayers as _default_multilayers,
    resolve_pretrained_reference,
    resolve_pretrained_path,
    tokens_to_output,
)
from .vendor.ijepa.src.models import vision_transformer as vit


class IJEPA(nn.Module):
    """
    Wrapper for I-JEPA vision transformer checkpoints released by Meta AI.

    The model definition is imported from the Semantic Correspondence
    Benchmarking repository (submods/ijepa). By default the wrapper returns
    the final block features, but multi-layer outputs can be enabled via
    ``return_multilayer=True``.
    """

    def __init__(
        self,
        model_type: str,
        checkpoint_path: Optional[str] = None,
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
    ) -> None:
        super().__init__()
        if output not in {"dense", "gap"}:
            raise ValueError("IJEPA backbone supports only 'dense' or 'gap' outputs.")
        self.supported_outputs = ("dense", "gap")
        self.default_global_output = "gap"
        self.supports_multilayer = True
        self.supports_layer_selection = True
        self.image_mean = "imagenet"

        if checkpoint_path is None:
            default_ckpt = resolve_pretrained_path("ijepa", f"{model_type}.pth.tar")
            checkpoint_path = str(default_ckpt)
        else:
            checkpoint_path = str(resolve_pretrained_reference(checkpoint_path))

        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"I-JEPA checkpoint was not found at {ckpt_path}. "
                "Provide checkpoint_path pointing to a *.pth.tar file."
            )

        if "h.16" in model_type:
            patch_size = 16
        else:
            patch_size = 14

        if "448" in model_type:
            pretrained_img_size = 448
        elif "224" in model_type:
            pretrained_img_size = 224
        else:
            pretrained_img_size = 224

        # The released checkpoints use vit_huge architectures.
        self.model = vit.vit_huge(patch_size=patch_size, img_size=[pretrained_img_size])

        state_dict = torch.load(ckpt_path, map_location="cpu")["encoder"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"I-JEPA checkpoint {ckpt_path} missing keys: {missing}"
            )
        if unexpected:
            raise RuntimeError(
                f"I-JEPA checkpoint {ckpt_path} has unexpected keys: {unexpected}"
            )

        self.model.eval().to(torch.float32)

        self.patch_size = self.model.patch_embed.patch_size
        self.output = output
        num_blocks = len(self.model.blocks)
        self.num_blocks = num_blocks
        self.return_multilayer = return_multilayer

        if return_multilayer:
            multilayers = _default_multilayers(num_blocks)
            self.multilayers = sorted(set(multilayers))
            self.feat_dim = [self.model.embed_dim] * len(self.multilayers)
            self.layer = "-".join(str(idx) for idx in self.multilayers)
        else:
            if layer == -1:
                layer = num_blocks - 1
            if layer < 0 or layer >= num_blocks:
                raise ValueError(
                    f"Requested layer {layer} outside valid range [0, {num_blocks - 1}]"
                )
            self.multilayers = [layer]
            self.feat_dim = self.model.embed_dim
            self.layer = str(layer)

        self.checkpoint_name = ckpt_path.stem

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = center_padding(images, self.patch_size)
        h_tokens = images.shape[-2] // self.patch_size
        w_tokens = images.shape[-1] // self.patch_size

        x = self.model.patch_embed(images)
        pos_embed = self._resize_pos_embed((h_tokens, w_tokens), x.device, x.dtype)
        x = x + pos_embed.expand(x.shape[0], -1, -1)

        selected: List[torch.Tensor] = []
        target_layers = set(self.multilayers)
        for idx, blk in enumerate(self.model.blocks):
            x = blk(x)
            if idx in target_layers:
                tokens = x
                if self.model.norm is not None and idx == self.num_blocks - 1:
                    tokens = self.model.norm(tokens)
                selected.append(tokens)
                if len(selected) == len(target_layers):
                    break

        if not selected:
            raise RuntimeError("IJEPA backbone did not capture any intermediate layers.")

        outputs: List[torch.Tensor] = []
        for tokens in selected:
            outputs.append(
                tokens_to_output(
                    self.output,
                    tokens,
                    None,
                    (h_tokens, w_tokens),
                )
            )

        return outputs if self.return_multilayer else outputs[0]

    def _resize_pos_embed(
        self, target_hw: tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos_embed = self.model.pos_embed
        if pos_embed.ndim == 2:
            pos_embed = pos_embed.unsqueeze(0)

        orig_tokens = pos_embed.shape[1]
        dim = pos_embed.shape[-1]
        orig_size = int(orig_tokens**0.5 + 0.5)

        pos = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(
            pos,
            size=target_hw,
            mode="bicubic",
            align_corners=False,
        )
        pos = pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return pos.to(device=device, dtype=dtype)
