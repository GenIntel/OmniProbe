
import torch
from torch import nn
from transformers import ViTMAEForPreTraining
from transformers.modeling_outputs import BaseModelOutput

from .utils import default_multilayers, get_2d_sincos_pos_embed, tokens_to_output


class MAE(nn.Module):
    def __init__(
        self,
        checkpoint="facebook/vit-mae-base",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        """Code based on transformer database"""
        super().__init__()

        assert output in ["cls", "gap", "dense"], "Options: [cls, gap, dense]"
        self.output = output

        self.checkpoint_name = checkpoint.split("/")[1]

        self.vit = ViTMAEForPreTraining.from_pretrained(checkpoint).vit
        self.vit = self.vit.eval()

        # resize pos embedding
        # resize embedding for new size
        patch_size = self.vit.config.patch_size
        self.patch_size = patch_size
        self.layer = layer

        self.image_size = self.vit.embeddings.patch_embeddings.image_size
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        feat_dim = self.vit.config.hidden_size
        num_layers = len(self.vit.encoder.layer)
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

    def resize_pos_embed(self, image_size):
        # ViT patch embedding uses floor(H / patch), floor(W / patch) when image
        # size is not divisible by patch size. Mirror that behavior here so MAE
        # can run on tasks such as AP10K (e.g., image_size=840 for patch_size=16).
        self.feat_h = image_size[0] // self.patch_size
        self.feat_w = image_size[1] // self.patch_size
        if self.feat_h <= 0 or self.feat_w <= 0:
            raise ValueError(
                f"Invalid image size {image_size} for patch size {self.patch_size}."
            )
        embed_dim = self.vit.config.hidden_size
        self.vit.embeddings.patch_embeddings.image_size = image_size
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.feat_h, self.feat_w), add_cls_token=True
        )
        device = self.vit.embeddings.patch_embeddings.projection.weight.device
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device),
            requires_grad=False,
        )

    def embed_forward(self, embedder, pixel_values):
        # No masking here ...
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = embedder.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + embedder.position_embeddings[:, 1:, :]

        # append cls token
        cls_token = embedder.cls_token + embedder.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings

    def forward_vit_encoder(self, hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=False,):

        hidden_states_list = []
        for i, layer_module in enumerate(self.vit.encoder.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # HuggingFace ViTMAELayer returns a tuple (hidden_states, ...);
            # extract the tensor to avoid passing a tuple on the next iteration.
            layer_out = layer_module(hidden_states, layer_head_mask)
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            hidden_states_list.append(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states, 
            hidden_states=hidden_states_list if output_hidden_states else None,
            attentions=None)
       

    def forward(self, images):
        # check if positional embeddings are correct
        if self.image_size != images.shape[-2:]:
            self.resize_pos_embed(images.shape[-2:])

        # from MAE implementation
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)

        # ---- hidden ----
        embedding_output = self.embed_forward(self.vit.embeddings, images)
        encoder_outputs = self.forward_vit_encoder(embedding_output,
            head_mask=head_mask,
            output_attentions=self.vit.config.output_attentions,
            output_hidden_states=True,
            return_dict=self.vit.config.return_dict,
        )

        outputs = []
        for layer_i in self.multilayers:
            x_i = encoder_outputs.hidden_states[layer_i]
            x_i = tokens_to_output(
                self.output, x_i[:, 1:], x_i[:, 0], (self.feat_h, self.feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
