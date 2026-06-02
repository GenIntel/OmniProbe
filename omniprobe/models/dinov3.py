import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from .utils import (
    center_padding,
    default_multilayers,
    resolve_pretrained_reference,
    tokens_to_output,
)



def _load_local_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(weights_path), device="cpu")

    state_dict = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state_dict, dict):
        for checkpoint_key in ("model", "teacher", "student", "state_dict", "module"):
            if checkpoint_key in state_dict and isinstance(state_dict[checkpoint_key], dict):
                state_dict = state_dict[checkpoint_key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected DINOv3 checkpoint payload type: {type(state_dict)!r}")
    return state_dict


def _is_hf_dinov3_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return "embeddings.cls_token" in state_dict or any(
        key.startswith("layer.0.") for key in state_dict
    )


def _convert_hf_dinov3_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted = {}

    def _match_token_shape(name: str, tensor: torch.Tensor) -> torch.Tensor:
        target = model_state_dict.get(name)
        if target is None or tensor.shape == target.shape:
            return tensor
        if tensor.ndim == target.ndim + 1 and tensor.shape[0] == 1:
            squeezed = tensor.squeeze(0)
            if squeezed.shape == target.shape:
                return squeezed
        if tensor.ndim == target.ndim + 1 and tensor.shape[1] == 1:
            squeezed = tensor.squeeze(1)
            if squeezed.shape == target.shape:
                return squeezed
        return tensor

    if "embeddings.cls_token" in state_dict:
        converted["cls_token"] = _match_token_shape(
            "cls_token",
            state_dict["embeddings.cls_token"],
        )
    if "embeddings.register_tokens" in state_dict:
        converted["storage_tokens"] = _match_token_shape(
            "storage_tokens",
            state_dict["embeddings.register_tokens"],
        )
    if "embeddings.mask_token" in state_dict:
        converted["mask_token"] = _match_token_shape(
            "mask_token",
            state_dict["embeddings.mask_token"],
        )
    if "embeddings.patch_embeddings.weight" in state_dict:
        converted["patch_embed.proj.weight"] = state_dict["embeddings.patch_embeddings.weight"]
    if "embeddings.patch_embeddings.bias" in state_dict:
        converted["patch_embed.proj.bias"] = state_dict["embeddings.patch_embeddings.bias"]
    if "norm.weight" in state_dict:
        converted["norm.weight"] = state_dict["norm.weight"]
    if "norm.bias" in state_dict:
        converted["norm.bias"] = state_dict["norm.bias"]

    layer_indices = sorted(
        {
            int(key.split(".", 2)[1])
            for key in state_dict
            if key.startswith("layer.") and key.split(".", 2)[1].isdigit()
        }
    )
    for idx in layer_indices:
        src = f"layer.{idx}."
        dst = f"blocks.{idx}."

        for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            key = src + suffix
            if key in state_dict:
                converted[dst + suffix] = state_dict[key]

        q_w = state_dict.get(src + "attention.q_proj.weight")
        k_w = state_dict.get(src + "attention.k_proj.weight")
        v_w = state_dict.get(src + "attention.v_proj.weight")
        if q_w is not None and k_w is not None and v_w is not None:
            converted[dst + "attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = state_dict.get(src + "attention.q_proj.bias")
        k_b = state_dict.get(src + "attention.k_proj.bias")
        v_b = state_dict.get(src + "attention.v_proj.bias")
        if q_b is not None and k_b is not None and v_b is not None:
            converted[dst + "attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        if src + "attention.o_proj.weight" in state_dict:
            converted[dst + "attn.proj.weight"] = state_dict[src + "attention.o_proj.weight"]
        if src + "attention.o_proj.bias" in state_dict:
            converted[dst + "attn.proj.bias"] = state_dict[src + "attention.o_proj.bias"]

        if src + "layer_scale1.lambda1" in state_dict:
            converted[dst + "ls1.gamma"] = state_dict[src + "layer_scale1.lambda1"]
        if src + "layer_scale2.lambda1" in state_dict:
            converted[dst + "ls2.gamma"] = state_dict[src + "layer_scale2.lambda1"]

        if src + "mlp.up_proj.weight" in state_dict:
            converted[dst + "mlp.fc1.weight"] = state_dict[src + "mlp.up_proj.weight"]
        if src + "mlp.up_proj.bias" in state_dict:
            converted[dst + "mlp.fc1.bias"] = state_dict[src + "mlp.up_proj.bias"]
        if src + "mlp.down_proj.weight" in state_dict:
            converted[dst + "mlp.fc2.weight"] = state_dict[src + "mlp.down_proj.weight"]
        if src + "mlp.down_proj.bias" in state_dict:
            converted[dst + "mlp.fc2.bias"] = state_dict[src + "mlp.down_proj.bias"]

    return converted


def _prepare_local_dinov3_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    supplemented = (
        _convert_hf_dinov3_state_dict(state_dict, model_state_dict)
        if _is_hf_dinov3_state_dict(state_dict)
        else dict(state_dict)
    )
    for key, value in model_state_dict.items():
        if key.endswith("attn.qkv.bias") and key not in supplemented:
            supplemented[key] = value.clone()
        elif key.endswith("attn.qkv.bias_mask") and key not in supplemented:
            supplemented[key] = value.clone()
        elif key == "rope_embed.periods" and key not in supplemented:
            supplemented[key] = value.clone()
    return supplemented


class DinoV3(nn.Module):
    VARIANTS = {
        "vits16": {"hub_fn": "dinov3_vits16", "feat_dim": 384, "patch_size": 16, "model_type": "vit"},
        "vits16plus": {"hub_fn": "dinov3_vits16plus", "feat_dim": 384, "patch_size": 16, "model_type": "vit"},
        "vitb16": {"hub_fn": "dinov3_vitb16", "feat_dim": 768, "patch_size": 16, "model_type": "vit"},
        "vitl16": {"hub_fn": "dinov3_vitl16", "feat_dim": 1024, "patch_size": 16, "model_type": "vit"},
        "vitl16plus": {"hub_fn": "dinov3_vitl16plus", "feat_dim": 1024, "patch_size": 16, "model_type": "vit"},
        "vith16plus": {"hub_fn": "dinov3_vith16plus", "feat_dim": 1280, "patch_size": 16, "model_type": "vit"},
        "vit7b16": {"hub_fn": "dinov3_vit7b16", "feat_dim": 4096, "patch_size": 16, "model_type": "vit"},
        "convnext_tiny": {"hub_fn": "dinov3_convnext_tiny", "feat_dim": 768, "patch_size": 16, "model_type": "convnext"},
        "convnext_small": {"hub_fn": "dinov3_convnext_small", "feat_dim": 768, "patch_size": 16, "model_type": "convnext"},
        "convnext_base": {"hub_fn": "dinov3_convnext_base", "feat_dim": 1024, "patch_size": 16, "model_type": "convnext"},
        "convnext_large": {"hub_fn": "dinov3_convnext_large", "feat_dim": 1536, "patch_size": 16, "model_type": "convnext"},
    }

    def __init__(
        self,
        arch: str = "vitb16",
        *,
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
        weights: str | os.PathLike[str] = "LVD1689M",
        pretrained: bool = True,
        check_hash: bool = False,
        extra_hub_kwargs: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__()
        if arch not in self.VARIANTS:
            raise ValueError(f"Unsupported DINOv3 architecture '{arch}'.")

        variant = self.VARIANTS[arch]
        if output not in {"cls", "gap", "dense"}:
            raise ValueError(f"Unsupported output type '{output}'.")

        weights_str = str(weights)
        local_weights_path: Path | None = None
        if os.path.isabs(weights_str) or weights_str.startswith(".") or "/" in weights_str:
            local_weights_path = resolve_pretrained_reference(weights_str)
            weights_str = str(local_weights_path)

        use_local_state_dict = pretrained and local_weights_path is not None
        hub_kwargs: dict = {
            "pretrained": False if use_local_state_dict else pretrained,
            "weights": weights_str,
            "check_hash": check_hash,
        }
        if extra_hub_kwargs:
            hub_kwargs.update(extra_hub_kwargs)

        hub_fn = variant["hub_fn"]
        model = torch.hub.load(
            "facebookresearch/dinov3",
            hub_fn,
            trust_repo=True,
            **hub_kwargs,
        )

        self.model = model.eval().to(torch.float32)
        if use_local_state_dict:
            raw_state_dict = _load_local_state_dict(local_weights_path)
            normalized_state_dict = _prepare_local_dinov3_state_dict(
                raw_state_dict,
                self.model.state_dict(),
            )
            load_result = self.model.load_state_dict(normalized_state_dict, strict=False)
            allowed_missing = {"rope_embed.periods"}
            missing = set(load_result.missing_keys) - allowed_missing
            unexpected = set(load_result.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    "DINOv3 local checkpoint could not be normalized cleanly. "
                    f"Missing keys: {sorted(missing)}; Unexpected keys: {sorted(unexpected)}"
                )
        self.model_type = variant["model_type"]
        self.output = output
        self.arch = arch
        if getattr(self.model, "patch_size", None) is None and variant["patch_size"] is not None:
            setattr(self.model, "patch_size", variant["patch_size"])
        self.patch_size = getattr(self.model, "patch_size", variant["patch_size"])

        feat_dim = int(variant["feat_dim"])
        self.base_feat_dim = feat_dim

        num_layers = getattr(self.model, "n_blocks", None)
        if num_layers is None:
            raise AttributeError(f"DINOv3 backbone '{arch}' does not expose 'n_blocks'.")

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

        weight_tag = (
            local_weights_path.stem
            if local_weights_path is not None
            else weights_str
        )
        self.checkpoint_name = f"dinov3_{arch}_{weight_tag}"
        self.layer = "-".join(str(idx) for idx in self.multilayers)

    def _gather_features(self, patches: torch.Tensor, cls_token: torch.Tensor):
        if patches.ndim == 4:
            feat_hw = (patches.shape[-2], patches.shape[-1])
            dense_tokens = patches.flatten(2).transpose(1, 2)
        elif patches.ndim == 3:
            feat_size = int(patches.shape[1] ** 0.5)
            feat_hw = (feat_size, feat_size)
            dense_tokens = patches
        else:
            raise ValueError(f"Unexpected patch tensor shape {patches.shape}.")
        return dense_tokens, cls_token, feat_hw

    def forward(self, images: torch.Tensor):
        if self.patch_size is not None:
            images = center_padding(images, self.patch_size)

        query: Sequence[int] = self.multilayers
        outputs = self.model.get_intermediate_layers(
            images,
            n=list(query),
            reshape=True,
            return_class_token=True,
            norm=True,
        )
        outputs = [outputs[0]] if len(query) == 1 else list(outputs)

        feats = []
        for patches, cls_token in outputs:
            dense_tokens, cls_token, feat_hw = self._gather_features(patches, cls_token)
            feat = tokens_to_output(self.output, dense_tokens, cls_token, feat_hw)
            feats.append(feat)

        return feats[0] if len(feats) == 1 else feats
