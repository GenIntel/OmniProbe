import torch
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from .utils import center_padding, default_multilayers, tokens_to_output

VIT_ARCHS = {"vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"}


class DinoV3(nn.Module):
    """DINOv3 ViT backbone, loaded from Hugging Face Hub.

    Only the plain ViT variants are supported: ConvNeXt variants are excluded
    because their original feature extraction upsamples and renormalizes
    internal stages in a way the plain Hugging Face model doesn't expose, and
    vitl16plus is excluded because it has no Hugging Face Hub repo (only a
    gated torch.hub download).
    """

    VARIANTS = VIT_ARCHS

    def __init__(
        self,
        arch: str = "vitb16",
        *,
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
        weights: str = "LVD1689M",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if arch not in VIT_ARCHS:
            raise ValueError(f"Unsupported DINOv3 architecture '{arch}'.")
        if output not in {"cls", "gap", "dense"}:
            raise ValueError(f"Unsupported output type '{output}'.")

        repo_id = f"facebook/dinov3-{arch}-pretrain-{weights.lower()}"
        if pretrained:
            self.model = DINOv3ViTModel.from_pretrained(repo_id)
        else:
            self.model = DINOv3ViTModel(DINOv3ViTConfig.from_pretrained(repo_id))
        self.model = self.model.eval()

        config = self.model.config
        self.patch_size = config.patch_size
        self.num_register_tokens = config.num_register_tokens
        self.output = output

        feat_dim = config.hidden_size
        num_layers = config.num_hidden_layers
        multilayers = default_multilayers(num_layers)
        if return_multilayer:
            self.multilayers = multilayers
            self.feat_dim = [feat_dim] * len(multilayers)
        else:
            target_layer = multilayers[-1] if layer == -1 else layer
            if not (0 <= target_layer < num_layers):
                raise ValueError(
                    f"Requested layer {target_layer} outside valid range [0, {num_layers - 1}]"
                )
            self.multilayers = [target_layer]
            self.feat_dim = feat_dim

        self.checkpoint_name = f"dinov3_{arch}_{weights}"
        self.layer = "-".join(str(idx) for idx in self.multilayers)

    def forward(self, images: torch.Tensor):
        images = center_padding(images, self.patch_size)
        h = images.shape[-2] // self.patch_size
        w = images.shape[-1] // self.patch_size

        hidden_states = self.model(images, output_hidden_states=True).hidden_states

        outputs = []
        for layer_idx in self.multilayers:
            tokens = hidden_states[layer_idx + 1]
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1 + self.num_register_tokens :]
            outputs.append(tokens_to_output(self.output, patch_tokens, cls_token, (h, w)))

        return outputs[0] if len(outputs) == 1 else outputs
