from pathlib import Path

import torch
import torch.nn as nn

from .utils import (
    center_padding,
    default_multilayers as _default_multilayers,
    resolve_pretrained_path,
    resolve_pretrained_reference,
    tokens_to_output,
)


class VJEPA2Backbone(nn.Module):
    def __init__(
        self,
        model_type: str = "vjepa2_vit_large",
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
        pretrained: bool = True,
        force_reload: bool = False,
        checkpoint_path: str | None = None,
        checkpoint_url: str | None = None,
    ) -> None:
        super().__init__()
        if output not in {"dense", "gap"}:
            raise ValueError("VJEPA2 backbone supports only 'dense' or 'gap' outputs.")

        model_specs = {
            "vjepa2_vit_large": {
                "hub_model_type": "vjepa2_vit_large",
                "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
                "checkpoint_key": "target_encoder",
                "strict": False,
            },
            "vjepa2_vit_huge": {
                "hub_model_type": "vjepa2_vit_huge",
                "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
                "checkpoint_key": "target_encoder",
                "strict": False,
            },
            "vjepa2_vit_giant": {
                "hub_model_type": "vjepa2_vit_giant",
                "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
                "checkpoint_key": "target_encoder",
                "strict": False,
            },
            "vjepa2_vit_giant_384": {
                "hub_model_type": "vjepa2_vit_giant_384",
                "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt",
                "checkpoint_key": "target_encoder",
                "strict": False,
            },
            "vjepa2_1_vit_base_384": {
                "hub_model_type": "vjepa2_1_vit_base_384",
                "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
                "checkpoint_key": "ema_encoder",
                "strict": True,
            },
            "vjepa2_1_vit_large_384": {
                "hub_model_type": "vjepa2_1_vit_large_384",
                "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
                "checkpoint_key": "ema_encoder",
                "strict": True,
            },
        }
        if model_type not in model_specs:
            supported = ", ".join(sorted(model_specs))
            raise ValueError(f"Unsupported VJEPA2 model_type '{model_type}'. Supported: {supported}")

        spec = model_specs[model_type]
        loaded = torch.hub.load(
            "facebookresearch/vjepa2",
            spec["hub_model_type"],
            source="github",
            pretrained=False,
            force_reload=force_reload,
            trust_repo=True,
        )

        if isinstance(loaded, (tuple, list)):
            self.encoder = loaded[0]
        else:
            self.encoder = loaded

        if pretrained:
            # Cacheable URL download: prefer the active cache ($TORCH_HOME/checkpoints),
            # otherwise fall back to the repo-local checkpoints/ directory.
            checkpoint_dir = resolve_pretrained_path("vjepa2", prefer_cache=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if checkpoint_path is not None:
                state_dict = torch.load(resolve_pretrained_reference(checkpoint_path), map_location="cpu")
            else:
                resolved_checkpoint_url = checkpoint_url or spec["checkpoint_url"]
                cached_checkpoint = checkpoint_dir / Path(resolved_checkpoint_url).name
                if cached_checkpoint.exists():
                    state_dict = torch.load(cached_checkpoint, map_location="cpu")
                else:
                    state_dict = torch.hub.load_state_dict_from_url(
                        resolved_checkpoint_url,
                        model_dir=str(checkpoint_dir),
                        map_location="cpu",
                    )
            encoder_state = self._clean_backbone_key(state_dict[spec["checkpoint_key"]])
            self.encoder.load_state_dict(encoder_state, strict=bool(spec["strict"]))

        self.encoder.eval().to(torch.float32)

        self.output = output
        self.checkpoint_name = model_type
        self.patch_size = int(getattr(self.encoder, "patch_size", 16))
        self.embed_dim = int(getattr(self.encoder, "embed_dim", self.encoder.num_features))

        num_layers = len(self.encoder.blocks)
        if return_multilayer:
            self.multilayers = _default_multilayers(num_layers)
            self.feat_dim = [self.embed_dim] * len(self.multilayers)
            self.layer = "-".join(str(idx) for idx in self.multilayers)
        else:
            target_layer = num_layers - 1 if layer == -1 else layer
            if target_layer < 0 or target_layer >= num_layers:
                raise ValueError(
                    f"Requested layer {target_layer} outside valid range [0, {num_layers - 1}]"
                )
            self.multilayers = [target_layer]
            self.feat_dim = self.embed_dim
            self.layer = str(target_layer)

        self.return_multilayer = return_multilayer
        self.encoder.out_layers = self.multilayers

    @staticmethod
    def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cleaned = {}
        for key, value in state_dict.items():
            key = key.replace("module.", "")
            key = key.replace("backbone.", "")
            cleaned[key] = value
        return cleaned

    def forward(self, images: torch.Tensor):
        if images.dim() not in {4, 5}:
            raise ValueError(f"Expected BCHW or BCTHW tensor, got shape {images.shape}")

        if images.dim() == 4:
            images = center_padding(images, self.patch_size)
            h_tokens = images.shape[-2] // self.patch_size
            w_tokens = images.shape[-1] // self.patch_size
            if getattr(self.encoder, "is_video", False) or self.checkpoint_name.startswith("vjepa2_1_"):
                images = images.unsqueeze(2)
            if getattr(self.encoder, "is_video", False):
                tubelet = int(getattr(self.encoder, "tubelet_size", 1))
                if tubelet > 1:
                    images = images.repeat(1, 1, tubelet, 1, 1)
        else:
            h_tokens = images.shape[-2] // self.patch_size
            w_tokens = images.shape[-1] // self.patch_size

        tokens = self.encoder(images)
        if not isinstance(tokens, list):
            tokens = [tokens]

        outputs = []
        for token_map in tokens:
            outputs.append(tokens_to_output(self.output, token_map, None, (h_tokens, w_tokens)))
        return outputs if self.return_multilayer else outputs[0]
