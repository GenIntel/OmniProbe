
import einops as E
import open_clip
import torch
from pathlib import Path
from torch import nn

from .utils import (
    center_padding,
    default_multilayers,
    resolve_pretrained_reference,
    resize_pos_embed,
    tokens_to_output,
)


class CLIP(nn.Module):
    def __init__(
        self,
        arch="ViT-B-16",
        checkpoint="openai",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        assert output in ["cls", "gap", "dense"]
        self.output = output
        resolved_checkpoint = checkpoint
        if "/" in checkpoint:
            candidate = resolve_pretrained_reference(checkpoint)
            if candidate is not None and candidate.exists():
                resolved_checkpoint = str(candidate)
        checkpoint_name = Path(resolved_checkpoint).stem if "/" in str(resolved_checkpoint) else resolved_checkpoint
        self.checkpoint_name = "clip_" + arch.replace("-", "").lower() + checkpoint_name

        # Initialize a pre-trained CLIP image encoder and freeze it.
        _clip_model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=resolved_checkpoint
        )
        _clip_model = _clip_model.eval().to(torch.float32)
        self.visual = _clip_model.visual
        del _clip_model

        # Extract some attributes from CLIP module for easy access.
        self.patch_size = self.visual.conv1.stride[0]

        # get feature dimension
        feat_dim = self.visual.transformer.width
        feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]

        # get extraction targets
        n_layers = len(self.visual.transformer.resblocks)
        multilayers = default_multilayers(n_layers)

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims[-1]
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # clip stuff
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        # concat cls token
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)

        # add pos embed
        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))

        embeds = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, _x in enumerate(embeds):
            _x = tokens_to_output(self.output, _x[:, 1:], _x[:, 0], out_hw)
            outputs.append(_x)

        return outputs[0] if len(outputs) == 1 else outputs
