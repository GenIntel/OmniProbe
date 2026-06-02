import math
import warnings
from typing import List

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import InterpolationMode, Normalize, Resize
from torchvision.transforms.functional import to_pil_image, to_tensor


if not hasattr(InterpolationMode, "NEAREST_EXACT"):
    InterpolationMode.NEAREST_EXACT = InterpolationMode.NEAREST


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img
    if img.ndim != 3:
        raise ValueError(f"Expected 3D tensor image, got shape {img.shape}")
    if img.shape[0] in (1, 3):
        tensor = img.detach().cpu()
    else:
        tensor = img.detach().cpu().permute(2, 0, 1)
    tensor = tensor.float().clamp(0.0, 1.0)
    return to_pil_image(tensor)


def _dtype_from_name(torch_dtype):
    if isinstance(torch_dtype, str):
        return getattr(torch, torch_dtype)
    return torch_dtype


def _load_kwargs(torch_dtype, use_flash_attention, cache_dir):
    kwargs = {"torch_dtype": _dtype_from_name(torch_dtype)}
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return kwargs


def _from_pretrained(model_cls, model_name, load_kwargs, trust_remote_code=True):
    try:
        return model_cls.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
    except ImportError as exc:
        if "flash_attn" not in str(exc) or "attn_implementation" not in load_kwargs:
            raise
        warnings.warn(
            "FlashAttention2 requested but not available; falling back to standard attention.",
            RuntimeWarning,
        )
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("attn_implementation")
        return model_cls.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **fallback_kwargs,
        )


def _tokens_to_maps(tokens, grids, hidden_size, spatial_merge):
    feature_maps = []
    for token, grid in zip(tokens, grids):
        grid_t, grid_h, grid_w = [int(val) for val in grid.tolist()]
        token_len = int(token.shape[0])
        if token_len == 0:
            raise ValueError("Received zero-length token sequence from visual encoder.")

        merge = max(1, int(spatial_merge))
        h = max(1, grid_h // merge)
        w = max(1, grid_w // merge)

        if h * w != token_len:
            merge_area = max(1, (grid_t * grid_h * grid_w) // max(1, token_len))
            merge = max(1, int(math.isqrt(merge_area)))
            h = max(1, grid_h // merge)
            w = max(1, grid_w // merge)

        if h * w != token_len:
            h = max(1, int(round(math.sqrt(token_len))))
            w = max(1, token_len // h)

        feat = token[: h * w].reshape(h, w, hidden_size)
        feature_maps.append(feat.permute(2, 0, 1).contiguous())
    return torch.stack(feature_maps, dim=0).to(torch.float32)


def _square_tokens_to_maps(tokens):
    batch_size, token_count, hidden_size = tokens.shape
    side = int(math.sqrt(token_count))
    if side * side != token_count:
        raise ValueError(f"Cannot reshape {token_count} visual tokens into a square feature map.")
    return tokens.reshape(batch_size, side, side, hidden_size).permute(0, 3, 1, 2).contiguous().to(torch.float32)


class QwenVLVisualBackbone(nn.Module):
    """
    Dense visual encoder wrapper for Qwen2-VL and Qwen3-VL checkpoints.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        model_family="qwen3",
        output="dense",
        return_multilayer=False,
        min_pixels=200704,
        max_pixels=1003520,
        torch_dtype="bfloat16",
        use_flash_attention=False,
        cache_dir=None,
        image_mean="raw",
    ):
        super().__init__()
        if return_multilayer:
            raise ValueError("QwenVLVisualBackbone does not support return_multilayer=True")
        if output not in ["dense", "gap"]:
            raise ValueError("Only dense or gap outputs are supported.")
        self.output = output
        self.image_mean = image_mean
        self.checkpoint_name = model_name.split("/")[-1]

        from transformers import AutoProcessor

        if model_family == "qwen2":
            from transformers import Qwen2VLForConditionalGeneration

            model_cls = Qwen2VLForConditionalGeneration
        elif model_family == "qwen2_5":
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration
        elif model_family == "qwen3":
            from transformers import Qwen3VLForConditionalGeneration

            model_cls = Qwen3VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported Qwen model_family '{model_family}'.")

        processor_kwargs = {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "trust_remote_code": True,
        }
        if cache_dir:
            processor_kwargs["cache_dir"] = cache_dir
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        model = _from_pretrained(
            model_cls,
            model_name,
            _load_kwargs(torch_dtype, use_flash_attention, cache_dir),
        )
        if not hasattr(model, "visual") and hasattr(model, "model"):
            self.visual = model.model.visual.eval()
        elif hasattr(model, "visual"):
            self.visual = model.visual.eval()
        else:
            raise AttributeError(f"The checkpoint '{model_name}' does not expose a visual encoder.")
        del model

        self.spatial_merge = int(getattr(self.visual, "spatial_merge_size", 1))
        hidden_size = getattr(self.visual.config, "out_hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.visual.config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Unable to infer Qwen visual hidden size.")
        self.hidden_size = int(hidden_size)
        self.feat_dim = self.hidden_size
        self.patch_size = int(getattr(self.visual, "patch_size", getattr(self.visual.config, "patch_size", 14)))
        self.layer = "last"

    def _prepare_inputs(self, images):
        pil_images: List[Image.Image] = [_to_pil(img) for img in images]
        return self.processor(
            images=pil_images,
            text=[""] * len(pil_images),
            padding=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def forward(self, images):
        if images.dim() != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {images.shape}")
        batch_inputs = self._prepare_inputs(images)
        device = next(self.visual.parameters()).device
        dtype = next(self.visual.parameters()).dtype
        pixel_values = batch_inputs["pixel_values"].to(device, dtype=dtype)
        image_grid_thw = batch_inputs["image_grid_thw"].to(device)
        visual_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        hidden_states = visual_outputs[0] if isinstance(visual_outputs, (tuple, list)) else visual_outputs
        split_sizes = (image_grid_thw.prod(dim=-1) // (self.spatial_merge ** 2)).tolist()
        tokens = torch.split(hidden_states, split_sizes, dim=0)
        feats = _tokens_to_maps(tokens, image_grid_thw, self.hidden_size, self.spatial_merge)
        if self.output == "gap":
            feats = feats.mean(dim=(2, 3))
        return feats


class InternVLVisualBackbone(nn.Module):
    """
    Dense visual encoder wrapper for HF-format InternVL checkpoints.
    """

    def __init__(
        self,
        model_name="OpenGVLab/InternVL3_5-8B-HF",
        output="dense",
        return_multilayer=False,
        torch_dtype="bfloat16",
        use_flash_attention=False,
        cache_dir=None,
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
        image_mean="raw",
    ):
        super().__init__()
        if return_multilayer:
            raise ValueError("InternVLVisualBackbone does not support return_multilayer=True")
        if output not in ["dense", "gap"]:
            raise ValueError("Only dense or gap outputs are supported.")
        self.output = output
        self.image_mean = image_mean
        self.checkpoint_name = model_name.split("/")[-1]
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

        from transformers import AutoProcessor, InternVLForConditionalGeneration

        processor_kwargs = {"trust_remote_code": True}
        if cache_dir:
            processor_kwargs["cache_dir"] = cache_dir
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        model = _from_pretrained(
            InternVLForConditionalGeneration,
            model_name,
            _load_kwargs(torch_dtype, use_flash_attention, cache_dir),
        )
        self.vision_model = model.model.vision_tower.eval()
        del model

        vision_config = self.vision_model.config
        self.feat_dim = int(vision_config.hidden_size)
        patch_size = getattr(vision_config, "patch_size", [14, 14])
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
        self.layer = str(vision_feature_layer)

    def _prepare_inputs(self, images):
        pil_images: List[Image.Image] = [_to_pil(img) for img in images]
        return self.processor.image_processor(images=pil_images, return_tensors="pt")

    @torch.no_grad()
    def forward(self, images):
        if images.dim() != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {images.shape}")
        batch_inputs = self._prepare_inputs(images)
        device = next(self.vision_model.parameters()).device
        dtype = next(self.vision_model.parameters()).dtype
        pixel_values = batch_inputs["pixel_values"].to(device, dtype=dtype)
        outputs = self.vision_model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        if self.vision_feature_select_strategy == "default" and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        feats = _square_tokens_to_maps(tokens)
        if self.output == "gap":
            feats = feats.mean(dim=(2, 3))
        return feats


class LlavaOneVisionVisualBackbone(nn.Module):
    """
    Dense visual tower wrapper for LLaVA-OneVision checkpoints.
    """

    def __init__(
        self,
        model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        output="dense",
        return_multilayer=False,
        torch_dtype="bfloat16",
        use_flash_attention=False,
        cache_dir=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        image_mean="raw",
    ):
        super().__init__()
        if return_multilayer:
            raise ValueError("LlavaOneVisionVisualBackbone does not support return_multilayer=True")
        if output not in ["dense", "gap"]:
            raise ValueError("Only dense or gap outputs are supported.")
        self.output = output
        self.image_mean = image_mean
        self.checkpoint_name = model_name.split("/")[-1]

        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        processor_kwargs = {"trust_remote_code": True}
        if cache_dir:
            processor_kwargs["cache_dir"] = cache_dir
        processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        model = _from_pretrained(
            LlavaOnevisionForConditionalGeneration,
            model_name,
            _load_kwargs(torch_dtype, use_flash_attention, cache_dir),
        )
        self.vision_tower = model.model.vision_tower.eval()
        self.config = model.config
        del model

        self.vision_feature_layer = (
            self.config.vision_feature_layer if vision_feature_layer is None else vision_feature_layer
        )
        self.vision_feature_select_strategy = (
            self.config.vision_feature_select_strategy
            if vision_feature_select_strategy is None
            else vision_feature_select_strategy
        )
        vision_config = self.config.vision_config
        self.feat_dim = int(vision_config.hidden_size)
        self.patch_size = int(getattr(vision_config, "patch_size", 14))
        self.image_size = int(getattr(vision_config, "image_size", 384))
        image_processor = processor.image_processor
        mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.resize = Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.normalize = Normalize(mean=mean, std=std)
        self.layer = str(self.vision_feature_layer)

    def _prepare_pixel_values(self, images):
        tensors = []
        for img in images:
            pil = _to_pil(img)
            tensor = self.normalize(to_tensor(self.resize(pil)))
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def forward(self, images):
        if images.dim() != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {images.shape}")
        device = next(self.vision_tower.parameters()).device
        dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = self._prepare_pixel_values(images).to(device, dtype=dtype)
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        if isinstance(self.vision_feature_layer, int):
            tokens = image_features.hidden_states[self.vision_feature_layer]
        else:
            hs_pool = [image_features.hidden_states[idx] for idx in self.vision_feature_layer]
            tokens = torch.cat(hs_pool, dim=-1)
        if self.vision_feature_select_strategy == "default":
            tokens = tokens[:, 1:, :]
        feats = _square_tokens_to_maps(tokens)
        if self.output == "gap":
            feats = feats.mean(dim=(2, 3))
        return feats
