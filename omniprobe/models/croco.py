
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .utils import (
    center_padding,
    default_multilayers,
    get_2d_sincos_pos_embed,
    resolve_pretrained_reference,
    resolve_pretrained_path,
    tokens_to_output,
)
from .vendor.croco.models.croco import CroCoNet as _CroCoNet
from .vendor.croco.models.croco_downstream import croco_args_from_ckpt


_CROCO_ENCODER_NO_HEAD_CLASS = None


def _get_croco_encoder_no_head_class():
    """Construct a CroCo encoder subclass without decoder/head."""
    global _CROCO_ENCODER_NO_HEAD_CLASS
    if _CROCO_ENCODER_NO_HEAD_CLASS is None:
        class CroCoDownstreamMonocularEncoderNoHead(_CroCoNet):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def _set_mask_generator(self, *args, **kwargs):
                return

            def _set_mask_token(self, *args, **kwargs):
                self.mask_token = None
                return

            def _set_decoder(self, *args, **kwargs):
                return

            def _set_prediction_head(self, *args, **kwargs):
                return

            def forward(self, img: torch.Tensor) -> torch.Tensor:
                out, _, _ = self._encode_image(img, do_mask=False)
                return out

        _CROCO_ENCODER_NO_HEAD_CLASS = CroCoDownstreamMonocularEncoderNoHead

    return _CROCO_ENCODER_NO_HEAD_CLASS


@dataclass
class CroCoConfig:
    checkpoint_path: Path
    pos_embed_override: Optional[str]
    stride: int
    output: str
    return_multilayer: bool
    layer: int


_default_multilayers = default_multilayers  # backwards compat alias


class CroCoBackbone(nn.Module):
    """
    Thin wrapper around the CroCo encoder for dense feature extraction.

    The implementation relies on the Semantic Correspondence Benchmarking
    repository for the CroCo model definition. We dynamically import the
    modules from that checkout and load the provided checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        pos_embed_override: Optional[str] = None,
        stride: int = 16,
        image_size: Union[int, Sequence[int]] = 960,
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
    ) -> None:
        super().__init__()
        if output not in {"dense", "gap"}:
            raise ValueError("CroCoBackbone supports only 'dense' or 'gap' outputs.")

        CroCoDownstreamMonocularEncoderNoHead = _get_croco_encoder_no_head_class()

        if checkpoint_path is None:
            default_ckpt = resolve_pretrained_path(
                "croco", "CroCo_V2_ViTBase_BaseDecoder.pth"
            )
            checkpoint_path = str(default_ckpt)
        else:
            checkpoint_path = str(resolve_pretrained_reference(checkpoint_path))

        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"CroCo checkpoint was not found at {ckpt_path}. "
                "Provide checkpoint_path pointing to a CroCo *.pth file."
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        croco_kwargs = croco_args_from_ckpt(ckpt)
        if pos_embed_override is not None:
            croco_kwargs["pos_embed"] = pos_embed_override

        if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
            size_seq = list(image_size)
            if len(size_seq) != 2:
                raise ValueError(
                    f"Expected image_size sequence of length 2, got {len(size_seq)}."
                )
            self.image_size = (int(size_seq[0]), int(size_seq[1]))
        else:
            self.image_size = (int(image_size), int(image_size))

        croco_kwargs.setdefault("img_size", self.image_size[0])

        self.model = CroCoDownstreamMonocularEncoderNoHead(**croco_kwargs)

        encoder_state = {
            key: value
            for key, value in ckpt["model"].items()
            if key.startswith("enc") or key.startswith("patch_embed.")
        }
        missing, unexpected = self.model.load_state_dict(
            encoder_state, strict=False
        )
        if missing:
            raise RuntimeError(
                f"CroCo checkpoint at {ckpt_path} is missing keys: {missing}"
            )
        if unexpected:
            raise RuntimeError(
                f"CroCo checkpoint at {ckpt_path} has unexpected keys: {unexpected}"
            )

        patch_size = self.model.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size

        # Optional stride adjustment.
        if stride != self.patch_size:
            self.model.patch_embed.proj.stride = (stride, stride)
        self.stride = stride

        grid_h = self.image_size[0] // self.patch_size
        grid_w = self.image_size[1] // self.patch_size
        self.model.patch_embed.img_size = self.image_size
        self.model.patch_embed.grid_size = (grid_h, grid_w)
        self.model.patch_embed.num_patches = grid_h * grid_w
        if hasattr(self.model, "mask_generator") and hasattr(
            self.model.mask_generator, "num_patches"
        ):
            self.model.mask_generator.num_patches = self.model.patch_embed.num_patches

        self.model.eval().to(torch.float32)

        self.output = output
        self.return_multilayer = return_multilayer

        num_layers = len(self.model.enc_blocks)
        multilayers = _default_multilayers(num_layers)
        if return_multilayer:
            self.multilayers = multilayers
            self.feat_dim = [self.model.enc_embed_dim] * len(multilayers)
        else:
            if layer == -1:
                layer = multilayers[-1]
            if layer < 0 or layer >= num_layers:
                raise ValueError(
                    f"Requested layer {layer} outside valid range [0, {num_layers - 1}]"
                )
            self.multilayers = [layer]
            self.feat_dim = self.model.enc_embed_dim

        self.layer = "-".join(str(i) for i in self.multilayers)
        self.checkpoint_name = ckpt_path.stem

    def _update_image_size(self, height: int, width: int) -> None:
        current_h, current_w = self.model.patch_embed.img_size
        if (height, width) == (current_h, current_w):
            return

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "CroCoBackbone expects dimensions divisible by the patch size "
                f"{self.patch_size}. Received {(height, width)}."
            )

        grid_size = (height // self.patch_size, width // self.patch_size)

        self.model.patch_embed.img_size = (height, width)
        self.model.patch_embed.grid_size = grid_size
        self.model.patch_embed.num_patches = grid_size[0] * grid_size[1]
        if hasattr(self.model.patch_embed, "position_getter"):
            self.model.patch_embed.position_getter.cache_positions.clear()

        mask_gen = getattr(self.model, "mask_generator", None)
        if mask_gen is not None and hasattr(mask_gen, "num_patches"):
            mask_ratio = getattr(mask_gen, "num_mask", 0) / max(
                mask_gen.num_patches, 1
            )
            mask_gen.num_patches = self.model.patch_embed.num_patches
            mask_gen.num_mask = int(round(mask_ratio * mask_gen.num_patches))

        def _set_pos_buffer(name: str, embed_dim: int) -> None:
            pos_tensor = getattr(self.model, name, None)
            if pos_tensor is None:
                return
            pos = get_2d_sincos_pos_embed(
                embed_dim, grid_size, add_cls_token=False
            )
            setattr(
                self.model,
                name,
                pos_tensor.new_tensor(pos, dtype=pos_tensor.dtype),
            )

        _set_pos_buffer("enc_pos_embed", self.model.enc_embed_dim)
        if hasattr(self.model, "dec_embed_dim"):
            _set_pos_buffer("dec_pos_embed", self.model.dec_embed_dim)

        self.image_size = (height, width)

    def _encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        if self.return_multilayer:
            all_feats, *_ = self.model._encode_image(
                images, do_mask=False, return_all_blocks=True
            )
            selected = [all_feats[idx] for idx in self.multilayers]
        else:
            feats, *_ = self.model._encode_image(
                images, do_mask=False, return_all_blocks=False
            )
            selected = [feats]
        return selected

    def forward(self, images: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        images = center_padding(images, self.patch_size)
        self._update_image_size(images.shape[-2], images.shape[-1])
        h_tokens = images.shape[-2] // self.stride
        w_tokens = images.shape[-1] // self.stride

        features = self._encode(images)

        outputs: List[torch.Tensor] = []
        for tokens in features:
            dense_tokens = tokens  # B x N x C
            dense = tokens_to_output(
                self.output, dense_tokens, cls_token=None, feat_hw=(h_tokens, w_tokens)
            )
            outputs.append(dense)

        return outputs if self.return_multilayer else outputs[0]
