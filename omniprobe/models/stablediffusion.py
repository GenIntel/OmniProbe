
from pathlib import Path

import torch
from torch.nn.functional import interpolate

from .vendor.dift_sd import SDFeaturizer


class DIFT(torch.nn.Module):
    def __init__(
        self,
        model_id="sd2-community/stable-diffusion-2-1",
        time_step=250,
        output="dense",
        layer=1,
        return_multilayer=False,
    ):
        super().__init__()
        assert output in ["gap", "dense"], "Only supports gap or dense output"

        self.output = output
        self.time_step = time_step
        model_name = Path(model_id).name if model_id.startswith("/") else model_id.split("/")[-1]
        self.checkpoint_name = f"{model_name}_noise-{time_step}"
        self.patch_size = 16
        self.dift = SDFeaturizer(model_id)
        self.up_ft_index = [0, 1, 2, 3]  # keep all the upblock feats
        assert layer in [-1, 0, 1, 2, 3]

        feat_dims = [1280, 1280, 640, 320]
        multilayers = [0, 1, 2, 3]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            layer = multilayers[-1] if layer == -1 else layer
            self.feat_dim = feat_dims[layer]
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images, categories=None, prompts=None):
        spatial = []
        batch_size = images.shape[0]

        # handle prompts
        assert categories is None or prompts is None, "Cannot be both"
        if categories:
            prompts = [f"a photo of a {_c}" for _c in categories]
        elif prompts is None:
            prompts = ["" for _ in range(batch_size)]

        assert len(prompts) == batch_size

        spatial = self.dift.forward(
            images, prompts=prompts, t=self.time_step, up_ft_index=self.up_ft_index
        )
        h, w = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
        spatial = [spatial[i] for i in self.multilayers]

        assert self.output in ["gap", "dense"]
        if self.output == "gap":
            spatial = [x.mean(dim=(2, 3)) for x in spatial]
        elif self.output == "dense":
            spatial = [interpolate(x.contiguous(), (h, w)) for x in spatial]

        return spatial[0] if len(spatial) == 1 else spatial
