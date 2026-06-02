from pathlib import Path

import einops as E
import torch
from torch import nn

from .utils import (
    center_padding,
    default_multilayers,
    resolve_pretrained_reference,
    resize_pos_embed,
    tokens_to_output,
)
from .vendor.metaclip.src.mini_clip.factory import create_model_and_transforms as _create_model_and_transforms


_ARCH_CONFIGS = {
    "vits16": ("ViT-S-16-worldwide@WorldWideCLIP", 384, 12, 16),
    "vitb16": ("ViT-B-16-worldwide@WorldWideCLIP", 768, 12, 16),
    "vitl14": ("ViT-L-14-worldwide@WorldWideCLIP", 1024, 24, 14),
}


class MetaCLIP(nn.Module):
    def __init__(
        self,
        arch="vitl14",
        checkpoint="",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        assert output in ["cls", "gap", "dense"]
        assert arch in _ARCH_CONFIGS, f"Unknown arch '{arch}', choose from {list(_ARCH_CONFIGS)}"
        self.output = output

        checkpoint_path = resolve_pretrained_reference(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MetaCLIP checkpoint not found at {checkpoint_path}.")

        model_name, feat_dim, depth, patch_size = _ARCH_CONFIGS[arch]
        self.checkpoint_name = f"metaclip2_{arch}_{checkpoint_path.stem}"

        clip_model, _, _ = _create_model_and_transforms(model_name, pretrained="")
        clip_model = clip_model.to(torch.float32)

        state = torch.load(str(checkpoint_path), map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {key.replace("module.", ""): value for key, value in state.items()}
        clip_model.load_state_dict(state, strict=False)

        self.visual = clip_model.visual.eval()
        del clip_model

        self.patch_size = patch_size
        feat_dims = [feat_dim] * 4

        multilayers = default_multilayers(depth)
        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            target_layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [target_layer]

        self.layer = "-".join(str(x) for x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        cls = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([cls.to(x.dtype), x], dim=1)

        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))
        x = x.permute(1, 0, 2)

        embeds = []
        for idx, block in enumerate(self.visual.transformer.resblocks):
            x = block(x)
            if idx in self.multilayers:
                embeds.append(x.permute(1, 0, 2))
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for features in embeds:
            outputs.append(
                tokens_to_output(
                    self.output,
                    features[:, 1:],
                    features[:, 0],
                    out_hw,
                )
            )
        return outputs[0] if len(outputs) == 1 else outputs
