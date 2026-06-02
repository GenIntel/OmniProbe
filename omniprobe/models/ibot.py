
from pathlib import Path
from urllib.request import urlretrieve

import torch

from .ibot_transformers import vit_base, vit_large, vit_small
from .utils import (
    center_padding,
    default_multilayers,
    resolve_pretrained_path,
    tokens_to_output,
)

BASE_URL = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot"


class iBOT(torch.nn.Module):
    def __init__(
        self, model_type="base", output="dense", layer=-1, return_multilayer=False
    ):
        super().__init__()
        assert output in ["gap", "dense", "cls"]
        self.output = output
        self.return_multilayer = return_multilayer

        model_dict = {
            "small": ("ibot_vits16", "vits_16/checkpoint_teacher.pth"),
            "base": ("ibot_vitb16", "vitb_16/checkpoint_teacher.pth"),
            "base_in22k": ("ibot_vitb16_in22k", "vitb_16_pt22k/checkpoint_student.pth"),
            "large": ("ibot_vitl16", "vitl_16/checkpoint_teacher.pth"),
            "large_in22k": ("ibot_vitl16_in22k", "vitl_16_pt22k/checkpoint_student.pth"),
        }

        assert model_type in model_dict

        # Download model checkpoint
        ckpt_name, ckpt_url_path = model_dict[model_type]
        ckpt_path = resolve_pretrained_path("ibot", f"{ckpt_name}.pth", prefer_cache=True)
        if not ckpt_path.exists():
            download_path = f"{BASE_URL}/{ckpt_url_path}"
            urlretrieve(download_path, ckpt_path)

        # load and cleanup state dict
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # instantiate model
        if "small" in model_type:
            model_fn = vit_small
            feat_dim = 384
        elif "base" in model_type:
            model_fn = vit_base
            feat_dim = 768
        else:
            model_fn = vit_large
            feat_dim = 1024

        vit = model_fn(patch_size=16, return_all_tokens=True)
        vit.load_state_dict(state_dict, strict=False)
        vit.eval()

        # set parameters
        self.vit = vit
        self.patch_size = 16
        self.checkpoint_name = ckpt_name

        num_layers = len(self.vit.blocks)
        multilayers = default_multilayers(num_layers)

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, 1:]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
