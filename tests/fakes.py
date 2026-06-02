import torch


class TinyDenseBackbone(torch.nn.Module):
    def __init__(self, output="dense", layer=-1, return_multilayer=False):
        super().__init__()
        self.output = output
        self.patch_size = 2
        self.checkpoint_name = "tiny-dense"
        self.supported_outputs = ("dense", "gap")
        self.default_global_output = "gap"
        self.supports_multilayer = True
        self.supports_layer_selection = True
        self.image_mean = "imagenet"
        self.multilayers = [0, 1] if return_multilayer else [0]
        self.layer = "-".join(str(value) for value in self.multilayers)
        if return_multilayer:
            self.feat_dim = [4, 4]
        else:
            self.feat_dim = 4

    def forward(self, images):
        batch_size = images.shape[0]
        feat = torch.ones(batch_size, 4, 2, 2, device=images.device)
        if self.output == "gap":
            pooled = feat.mean(dim=(-2, -1))
            if isinstance(self.feat_dim, list):
                return [pooled, pooled]
            return pooled
        if isinstance(self.feat_dim, list):
            return [feat, feat]
        return feat


class TinyGlobalBackbone(torch.nn.Module):
    def __init__(self, output="cls", layer=-1, return_multilayer=False):
        super().__init__()
        self.output = output
        self.patch_size = 2
        self.checkpoint_name = "tiny-global"
        self.supported_outputs = ("cls", "gap")
        self.default_global_output = "cls"
        self.supports_multilayer = False
        self.supports_layer_selection = True
        self.image_mean = "imagenet"
        self.layer = "0"
        self.feat_dim = 4

    def forward(self, images):
        batch_size = images.shape[0]
        return torch.ones(batch_size, 4, device=images.device)
