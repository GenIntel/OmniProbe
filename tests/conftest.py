"""Shared pytest fixtures for omniprobe tests."""
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock


class _FakeDinoViT(nn.Module):
    """
    Minimal stand-in for a hub-loaded DINOv1/DINOv2/DUNE/VJEPA2 ViT.

    Supports the subset of the ViT interface that DINO, DINO_REG, DUNE, and
    VJEPA2 wrappers access: patch_embed, blocks, prepare_tokens_with_masks,
    prepare_tokens, get_intermediate_layers, and a generic forward.
    """

    def __init__(self, embed_dim=768, num_blocks=12, patch_size=14):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Structural attributes read in DINO.__init__
        _proj = MagicMock()
        _proj.kernel_size = (patch_size, patch_size)
        _patch_embed = MagicMock()
        _patch_embed.proj = _proj
        _patch_embed.patch_size = (patch_size, patch_size)
        self.patch_embed = _patch_embed

        # blocks: each is an Identity so x passes through unchanged
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        # num_features alias used by some wrappers (e.g. VJEPA2)
        self.num_features = embed_dim

    def _seq(self, images):
        """Return a (B, 1+h*w, D) sequence tensor."""
        B = images.shape[0]
        H, W = images.shape[-2], images.shape[-1]
        h = H // self.patch_size
        w = W // self.patch_size
        return torch.zeros(B, 1 + h * w, self.embed_dim)

    def prepare_tokens_with_masks(self, images, masks=None):
        return self._seq(images)

    def prepare_tokens(self, images):
        return self._seq(images)

    def get_intermediate_layers(self, x, n=1, return_class_token=False, reshape=False, **kwargs):
        B = x.shape[0]
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        if reshape:
            feat = torch.zeros(B, self.embed_dim, h, w)
        else:
            feat = torch.zeros(B, h * w, self.embed_dim)
        cls = torch.zeros(B, self.embed_dim)
        count = len(n) if isinstance(n, (list, tuple)) else n
        if return_class_token:
            return [(feat, cls)] * count
        return [feat] * count

    def load_state_dict(self, state_dict, strict=True):
        return [], []

    def forward(self, x):
        """Generic forward returning patch tokens (no CLS). Used by VJEPA2."""
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        h = H // self.patch_size
        w = W // self.patch_size
        feat = torch.zeros(B, h * w, self.embed_dim)
        out_layers = getattr(self, "out_layers", None)
        if out_layers is not None and len(out_layers) > 1:
            return [feat] * len(out_layers)
        return feat

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def fake_dino_hub(monkeypatch):
    """Patch torch.hub.load to return a tiny DINOv2-compatible ViT.

    Use this fixture in tests that instantiate DINO, DINO_REG, DUNE, or
    VJEPA2 backbones without needing real model weights or network access.
    """
    fake_vit = _FakeDinoViT()

    def _fake_load(repo, *args, **kwargs):
        return fake_vit

    monkeypatch.setattr("torch.hub.load", _fake_load)
    return fake_vit


# ---------------------------------------------------------------------------
# RADIO / C-RADIO fake model
# ---------------------------------------------------------------------------

class _FakeRADIOOutput:
    def __init__(self, features, summary):
        self.features = features
        self.summary = summary


class _FakeRADIOPatchGenerator:
    def __init__(self, patch_size=14, embed_dim=768):
        self.patch_size = patch_size
        self.num_skip = 1
        self._embed_dim = embed_dim

    def __call__(self, x):
        B = x.shape[0]
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        return torch.zeros(B, 1 + h * w, self._embed_dim)


class _FakeRADIOInnerModel(nn.Module):
    def __init__(self, embed_dim=768, patch_size=14, num_blocks=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_generator = _FakeRADIOPatchGenerator(patch_size, embed_dim)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.norm = nn.Identity()


class _FakeRADIOModel(nn.Module):
    """Minimal stand-in for a hub-loaded RADIO/C-RADIO model."""

    def __init__(self, embed_dim=768, patch_size=14, num_blocks=12):
        super().__init__()
        self.model = _FakeRADIOInnerModel(embed_dim, patch_size, num_blocks)

    def make_preprocessor_external(self):
        return lambda x: x

    def get_nearest_supported_resolution(self, h, w):
        return h, w

    def __call__(self, x, feature_fmt="NCHW"):
        B = x.shape[0]
        h = x.shape[-2] // self.model.patch_generator.patch_size
        w = x.shape[-1] // self.model.patch_generator.patch_size
        features = torch.zeros(B, self.model.embed_dim, h, w)
        summary = torch.zeros(B, self.model.embed_dim)
        return _FakeRADIOOutput(features, summary)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def fake_radio_hub(monkeypatch):
    """Patch torch.hub.load to return a tiny RADIO-compatible model."""
    fake_model = _FakeRADIOModel()

    def _fake_load(repo, *args, **kwargs):
        return fake_model

    monkeypatch.setattr("torch.hub.load", _fake_load)
    return fake_model


# ---------------------------------------------------------------------------
# CLIP fake model (open_clip)
# ---------------------------------------------------------------------------

class _FakeCLIPConv1(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.stride = (patch_size, patch_size)
        self._embed_dim = embed_dim
        self._patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        h = x.shape[-2] // self._patch_size
        w = x.shape[-1] // self._patch_size
        return torch.zeros(B, self._embed_dim, h, w)


class _FakeCLIPResBlock(nn.Module):
    def forward(self, x):
        return x


class _FakeCLIPTransformer:
    def __init__(self, width, num_blocks):
        self.width = width
        self.resblocks = [_FakeCLIPResBlock() for _ in range(num_blocks)]


class _FakeCLIPVisual(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, num_blocks=12, img_size=224):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.conv1 = _FakeCLIPConv1(embed_dim, patch_size)
        self.class_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1 + n_patches, embed_dim))
        self.ln_pre = nn.Identity()
        self.transformer = _FakeCLIPTransformer(embed_dim, num_blocks)

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


class _FakeCLIPModel:
    def __init__(self, embed_dim=768, patch_size=16, num_blocks=12, img_size=224):
        self.visual = _FakeCLIPVisual(embed_dim, patch_size, num_blocks, img_size)

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self

    def load_state_dict(self, state_dict, strict=False):
        return [], []


@pytest.fixture
def fake_openclip(monkeypatch):
    """Patch open_clip.create_model_and_transforms to return a tiny CLIP-compatible model."""
    fake = _FakeCLIPModel(embed_dim=768, patch_size=16, num_blocks=12)
    monkeypatch.setattr(
        "open_clip.create_model_and_transforms",
        lambda *a, **kw: (fake, None, None),
    )
    return fake


# ---------------------------------------------------------------------------
# MAE fake model (HuggingFace ViTMAEForPreTraining)
# ---------------------------------------------------------------------------

class _FakeMAEConfig:
    def __init__(self, patch_size=16, hidden_size=768, num_hidden_layers=12):
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_attentions = False
        self.return_dict = False


class _FakeMAEPatchEmbeddings(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=(224, 224)):
        super().__init__()
        self.image_size = tuple(img_size)
        self._embed_dim = embed_dim
        self._patch_size = patch_size
        # projection.weight.device is accessed in MAE.resize_pos_embed
        self.projection = nn.Linear(1, embed_dim, bias=False)

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        h = pixel_values.shape[-2] // self._patch_size
        w = pixel_values.shape[-1] // self._patch_size
        return torch.zeros(B, h * w, self._embed_dim)


class _FakeMAEEmbeddings(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=(224, 224)):
        super().__init__()
        h = img_size[0] // patch_size
        w = img_size[1] // patch_size
        self.patch_embeddings = _FakeMAEPatchEmbeddings(embed_dim, patch_size, img_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + h * w, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


class _FakeMAELayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


class _FakeMAEEncoder(nn.Module):
    def __init__(self, num_layers=12):
        super().__init__()
        self.layer = nn.ModuleList([_FakeMAELayer() for _ in range(num_layers)])


class _FakeMAEVit(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, num_layers=12, img_size=(224, 224)):
        super().__init__()
        self.config = _FakeMAEConfig(patch_size, embed_dim, num_layers)
        self.embeddings = _FakeMAEEmbeddings(embed_dim, patch_size, img_size)
        self.encoder = _FakeMAEEncoder(num_layers)

    def get_head_mask(self, head_mask, num_layers):
        return [None] * num_layers

    def eval(self):
        return self


class _FakeMAEPretraining:
    def __init__(self):
        self.vit = _FakeMAEVit()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def eval(self):
        return self


@pytest.fixture
def fake_hf_mae(monkeypatch):
    """Patch ViTMAEForPreTraining.from_pretrained to return a tiny MAE-compatible model."""
    monkeypatch.setattr("omniprobe.models.mae.ViTMAEForPreTraining", _FakeMAEPretraining)


# ---------------------------------------------------------------------------
# DeiT fake model (deit_utils factory functions)
# ---------------------------------------------------------------------------

class _FakeDeiTPatchEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.strict_img_size = True
        self._embed_dim = embed_dim
        self._patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        h = x.shape[-2] // self._patch_size
        w = x.shape[-1] // self._patch_size
        return torch.zeros(B, h * w, self._embed_dim)


class _FakeDeiTViT(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=384, num_blocks=12):
        super().__init__()
        h = w = img_size // patch_size
        self.num_features = embed_dim
        self.patch_embed = _FakeDeiTPatchEmbed(embed_dim, patch_size)
        # DeiT pos_embed has NO cls token (cls is prepended separately)
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_deit(monkeypatch):
    """Patch deit_base/large_patch16_LS factory functions with tiny fake ViTs."""
    base_vit = _FakeDeiTViT(embed_dim=768, patch_size=16, img_size=384, num_blocks=12)
    large_vit = _FakeDeiTViT(embed_dim=1024, patch_size=16, img_size=384, num_blocks=24)
    monkeypatch.setattr("omniprobe.models.deit.deit_base_patch16_LS",
                        lambda *a, **kw: base_vit)
    monkeypatch.setattr("omniprobe.models.deit.deit_large_patch16_LS",
                        lambda *a, **kw: large_vit)
    return base_vit


# ---------------------------------------------------------------------------
# SigLIP fake model (timm)
# ---------------------------------------------------------------------------

class _FakeSigLIPPatchEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size, img_size=384):
        super().__init__()
        h = w = img_size // patch_size
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (h, w)
        self.strict_img_size = True
        self._embed_dim = embed_dim
        self._patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        h = x.shape[-2] // self._patch_size
        w = x.shape[-1] // self._patch_size
        return torch.zeros(B, h * w, self._embed_dim)


class _FakeSigLIPViT(nn.Module):
    def __init__(self, embed_dim=1024, patch_size=16, img_size=384, num_blocks=24):
        super().__init__()
        h = w = img_size // patch_size
        self.patch_embed = _FakeSigLIPPatchEmbed(embed_dim, patch_size, img_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, embed_dim))
        self.num_features = embed_dim
        self.num_prefix_tokens = 0
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()

    def _pos_embed(self, x):
        return x + self.pos_embed

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_timm_siglip(monkeypatch):
    """Patch timm.create_model to return a tiny SigLIP-compatible ViT."""
    fake_vit = _FakeSigLIPViT(embed_dim=1024, patch_size=16, img_size=384, num_blocks=24)
    monkeypatch.setattr("omniprobe.models.siglip.timm.create_model",
                        lambda *a, **kw: fake_vit)
    return fake_vit


# ---------------------------------------------------------------------------
# iBOT fixture (reuses _FakeDinoViT for the ViT model)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ibot(monkeypatch, tmp_path):
    """Patch iBOT's checkpoint loading and vit_* factories with tiny fake models."""
    ckpt_file = tmp_path / "ibot_vitb16.pth"
    ckpt_file.write_bytes(b"")  # non-empty so Path.exists() returns True

    import torch as _torch
    monkeypatch.setattr(_torch, "load", lambda *a, **kw: {})
    monkeypatch.setattr("omniprobe.models.ibot.resolve_pretrained_path",
                        lambda *a, **kw: ckpt_file)
    monkeypatch.setattr("omniprobe.models.ibot.vit_base",
                        lambda patch_size=16, **kw: _FakeDinoViT(embed_dim=768, patch_size=patch_size))
    monkeypatch.setattr("omniprobe.models.ibot.vit_small",
                        lambda patch_size=16, **kw: _FakeDinoViT(embed_dim=384, patch_size=patch_size))
    monkeypatch.setattr("omniprobe.models.ibot.vit_large",
                        lambda patch_size=16, **kw: _FakeDinoViT(embed_dim=1024, patch_size=patch_size))


# ---------------------------------------------------------------------------
# SAM fake model (segment_anything registry)
# ---------------------------------------------------------------------------

class _FakeSAMNeckEntry:
    def __init__(self, in_channels):
        self.in_channels = in_channels


class _FakeSAMPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, embed_dim=256):
        super().__init__()
        self.proj = MagicMock()
        self.proj.kernel_size = [patch_size, patch_size]
        self._embed_dim = embed_dim
        self._patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        h = x.shape[-2] // self._patch_size
        w = x.shape[-1] // self._patch_size
        return torch.zeros(B, h, w, self._embed_dim)  # SAM uses BHWD format


class _FakeSAMImageEncoder(nn.Module):
    def __init__(self, feat_dim=256, patch_size=16, emb_h=14, emb_w=14, num_blocks=12):
        super().__init__()
        self.neck = [_FakeSAMNeckEntry(feat_dim)]
        self.pos_embed = nn.Parameter(torch.zeros(1, emb_h, emb_w, feat_dim))
        self.patch_embed = _FakeSAMPatchEmbed(patch_size, feat_dim)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


class _FakeSAMModel:
    def __init__(self, feat_dim=256, patch_size=16, emb_h=14, emb_w=14, num_blocks=12):
        self.image_encoder = _FakeSAMImageEncoder(feat_dim, patch_size, emb_h, emb_w, num_blocks)


_FAKE_SAM_REGISTRY = {
    "vit_b": lambda checkpoint=None: _FakeSAMModel(feat_dim=256, patch_size=16,
                                                    emb_h=14, emb_w=14, num_blocks=12),
    "vit_l": lambda checkpoint=None: _FakeSAMModel(feat_dim=256, patch_size=16,
                                                    emb_h=14, emb_w=14, num_blocks=24),
    "vit_h": lambda checkpoint=None: _FakeSAMModel(feat_dim=256, patch_size=16,
                                                    emb_h=14, emb_w=14, num_blocks=32),
}


@pytest.fixture
def fake_sam(monkeypatch, tmp_path):
    """Patch SAM registry and checkpoint path with tiny fake image encoder."""
    ckpt_file = tmp_path / "sam_vit_b_01ec64.pth"
    ckpt_file.write_bytes(b"")
    monkeypatch.setattr("omniprobe.models.sam.resolve_pretrained_path",
                        lambda *a, **kw: ckpt_file)
    monkeypatch.setattr("omniprobe.models.sam.sam_model_registry", _FAKE_SAM_REGISTRY)


# ---------------------------------------------------------------------------
# DINOv3 fake model (dinov3.hub.backbones factory)
# ---------------------------------------------------------------------------

class _FakeDinoV3Model(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, n_blocks=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_blocks = n_blocks

    def get_intermediate_layers(self, x, n, reshape=False, return_class_token=False, norm=False):
        B = x.shape[0]
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        count = len(n) if isinstance(n, (list, tuple)) else n
        if reshape:
            patches = torch.zeros(B, self.embed_dim, h, w)
        else:
            patches = torch.zeros(B, h * w, self.embed_dim)
        cls = torch.zeros(B, self.embed_dim)
        if return_class_token:
            return [(patches, cls)] * count
        return [patches] * count

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_dinov3(monkeypatch):
    """Patch torch.hub.load to return a tiny fake DINOv3 model."""
    fake = _FakeDinoV3Model(embed_dim=768, patch_size=16, n_blocks=12)
    monkeypatch.setattr("torch.hub.load", lambda *a, **kw: fake)
    return fake


# ---------------------------------------------------------------------------
# PIXIO fake model (pixio module factory functions)
# ---------------------------------------------------------------------------

class _FakePIXIOModel(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, images, block_ids=None):
        B = images.shape[0]
        h = images.shape[-2] // self.patch_size
        w = images.shape[-1] // self.patch_size
        N = h * w
        count = len(block_ids) if block_ids is not None else 1
        return [
            {
                "patch_tokens_norm": torch.zeros(B, N, self.embed_dim),
                "cls_tokens_norm": torch.zeros(B, 8, self.embed_dim),
            }
            for _ in range(count)
        ]

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_pixio(monkeypatch):
    """Patch pixio factory functions with a tiny fake PIXIO model."""
    import omniprobe.models.pixio as _pixio_wrapper

    fake = _FakePIXIOModel(embed_dim=768, patch_size=16)
    for fn_name in ["pixio_vitb16", "pixio_vitl16", "pixio_vith16",
                    "pixio_vit1b16", "pixio_vit5b16"]:
        if hasattr(_pixio_wrapper, fn_name):
            monkeypatch.setattr(_pixio_wrapper, fn_name, lambda pretrained=None: fake)
    return fake


# ---------------------------------------------------------------------------
# IJEPA fake model (src.models.vision_transformer.vit_huge)
# ---------------------------------------------------------------------------

class _FakeIJEPAPatchEmbed(nn.Module):
    def __init__(self, embed_dim=1280, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self._embed_dim = embed_dim
        self._patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        h = x.shape[-2] // self._patch_size
        w = x.shape[-1] // self._patch_size
        return torch.zeros(B, h * w, self._embed_dim)


class _FakeIJEPAViT(nn.Module):
    def __init__(self, embed_dim=1280, patch_size=16, img_size=224, num_blocks=32):
        super().__init__()
        h = w = img_size // patch_size
        self.embed_dim = embed_dim
        self.patch_embed = _FakeIJEPAPatchEmbed(embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, embed_dim))
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)

    def load_state_dict(self, state_dict, strict=True):
        # Return empty missing/unexpected so IJEPA's RuntimeError checks pass
        return [], []

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_ijepa(monkeypatch, tmp_path):
    """Patch IJEPA's vit_huge factory, checkpoint path, and torch.load."""
    from omniprobe.models.vendor.ijepa.src.models import vision_transformer as _vit_mod

    fake_model = _FakeIJEPAViT(embed_dim=1280, patch_size=16, img_size=224, num_blocks=32)
    monkeypatch.setattr(_vit_mod, "vit_huge", lambda **kw: fake_model)

    ckpt_file = tmp_path / "fake_ijepa.pth.tar"
    ckpt_file.write_bytes(b"")

    import torch as _torch
    monkeypatch.setattr(_torch, "load", lambda *a, **kw: {"encoder": {}})
    monkeypatch.setattr("omniprobe.models.ijepa.resolve_pretrained_path",
                        lambda *a, **kw: ckpt_file)
    return fake_model

# ---------------------------------------------------------------------------
# ConvNext fake model (open_clip / timm)
# ---------------------------------------------------------------------------

class _FakeConvNextBlock(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        return x


class _FakeConvNextStage(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeConvNextBlock(feat_dim)])

    def forward(self, x):
        return x


class _FakeConvNextTrunk(nn.Module):
    """Fake ConvNeXt trunk with stem and 4 stages at uniform dim=128."""

    FEAT_DIM = 128

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, self.FEAT_DIM, kernel_size=4, stride=4)
        self.stages = nn.ModuleList([_FakeConvNextStage(self.FEAT_DIM) for _ in range(4)])


class _FakeConvNextVisual:
    def __init__(self):
        self.trunk = _FakeConvNextTrunk()


class _FakeConvNextCLIPModel:
    def __init__(self):
        self.visual = _FakeConvNextVisual()

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_convnext(monkeypatch):
    """Patch open_clip and timm for ConvNext backbone tests."""
    fake_clip = _FakeConvNextCLIPModel()
    monkeypatch.setattr(
        "omniprobe.models.convnext.open_clip.create_model_and_transforms",
        lambda *a, **kw: (fake_clip, None, None),
    )
    monkeypatch.setattr(
        "omniprobe.models.convnext.timm.create_model",
        lambda *a, **kw: fake_clip.visual.trunk,
    )
    return fake_clip


# ---------------------------------------------------------------------------
# MiDaS / BEiT fake model (midas=True path via torch.hub.load)
# ---------------------------------------------------------------------------

class _FakeMiDaSPatchEmbed(nn.Module):
    """Fake patch embedding for midas_forward injection."""

    def __init__(self, embed_dim=1024, patch_size=16):
        super().__init__()
        self._embed_dim = embed_dim
        self._patch_size = patch_size
        self.patch_size = patch_size
        self.img_size = (384, 384)

    def forward(self, x):
        B = x.shape[0]
        h = x.shape[-2] // self._patch_size
        w = x.shape[-1] // self._patch_size
        return torch.zeros(B, h * w, self._embed_dim)


class _FakeMiDaSViT(nn.Module):
    """Fake ViT suitable for make_beit_backbone's midas_forward injection."""

    def __init__(self, embed_dim=1024, patch_size=16, num_blocks=24):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        h = w = 384 // patch_size

        self.patch_embed = _FakeMiDaSPatchEmbed(embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + h * w, embed_dim))
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.image_size = (384, 384)

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


class _FakeMiDaSPretrainedWrapper:
    def __init__(self, vit):
        self.model = vit


class _FakeDPTLargeModel:
    def __init__(self):
        vit = _FakeMiDaSViT()
        self.pretrained = _FakeMiDaSPretrainedWrapper(vit)


@pytest.fixture
def fake_midas_hub(monkeypatch):
    """Patch torch.hub.load and timm for make_beit_backbone tests."""
    fake_dpt = _FakeDPTLargeModel()
    monkeypatch.setattr("torch.hub.load", lambda *a, **kw: fake_dpt)
    monkeypatch.setattr(
        "omniprobe.models.midas_final.timm.create_model",
        lambda *a, **kw: fake_dpt.pretrained.model,
    )
    return fake_dpt


# ---------------------------------------------------------------------------
# DIY-SC fixture (DINO hub + fake AggregationNetwork via hub)
# ---------------------------------------------------------------------------

class _FakeDIYSCAggreNet(nn.Module):
    """Pass-through stand-in for the DIY-SC AggregationNetwork."""

    def forward(self, x):
        return x

    def eval(self):
        return self


@pytest.fixture
def fake_diy_sc(monkeypatch):
    """Patch torch.hub.load: DINO repos → fake ViT, DIY-SC repo → fake aggre_net."""
    fake_vit = _FakeDinoViT(embed_dim=768, num_blocks=12, patch_size=14)
    fake_aggre = _FakeDIYSCAggreNet()

    def _fake_load(repo, *args, **kwargs):
        repo_str = str(repo)
        if "odunkel" in repo_str or "DIY-SC" in repo_str:
            return fake_aggre
        return fake_vit

    monkeypatch.setattr("torch.hub.load", _fake_load)
    return fake_vit


# ---------------------------------------------------------------------------
# CroCo fake encoder (avoids loading real CroCo weights)
# ---------------------------------------------------------------------------

class _FakeCroCoEncoder(nn.Module):
    """Minimal stand-in for CroCoDownstreamMonocularEncoderNoHead."""

    def __init__(self, enc_embed_dim=768, enc_depth=12, patch_size=16, **kwargs):
        super().__init__()
        self.enc_embed_dim = enc_embed_dim
        h = w = 224 // patch_size

        self.patch_embed = MagicMock()
        self.patch_embed.patch_size = (patch_size, patch_size)
        self.patch_embed.img_size = (224, 224)
        self.patch_embed.grid_size = (h, w)
        self.patch_embed.num_patches = h * w

        self.enc_blocks = nn.ModuleList([nn.Identity() for _ in range(enc_depth)])
        # Registered as a plain tensor (not a Parameter) so resize_pos_embed can update it
        self.enc_pos_embed = torch.zeros(h * w, enc_embed_dim)

    def load_state_dict(self, state_dict, strict=True):
        return [], []

    def _encode_image(self, img, do_mask=False, return_all_blocks=False):
        B = img.shape[0]
        h = img.shape[-2] // 16
        w = img.shape[-1] // 16
        feat = torch.zeros(B, h * w, self.enc_embed_dim)
        if return_all_blocks:
            return [feat] * len(self.enc_blocks), None, None
        return feat, None, None

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_croco(monkeypatch, tmp_path):
    """Patch CroCo's encoder class and checkpoint loading.

    CroCo's submodule uses an absolute 'models.*' import that conflicts with
    other submodules.  We sidestep this entirely by injecting
    fake modules into sys.modules before the real croco imports can run, and
    by mocking _get_croco_encoder_no_head_class to return our fake encoder.
    """
    import sys
    from unittest.mock import MagicMock
    from pathlib import Path

    # --- Inject stub modules so `from submods.croco.models.croco_downstream
    #     import croco_args_from_ckpt` succeeds without importing real code ---
    def _fake_croco_args_from_ckpt(ckpt):
        return {}

    fake_croco_downstream = MagicMock()
    fake_croco_downstream.croco_args_from_ckpt = _fake_croco_args_from_ckpt

    # Register under the dotted name CroCoBackbone's dynamic import uses.
    # Use monkeypatch.setitem so pytest restores the originals after the test,
    # preventing MagicMock stubs from leaking into test_backbone_instantiation.py.
    _sentinel = object()
    for mod_name in [
        "submods",
        "submods.croco",
        "submods.croco.models",
        "submods.croco.models.croco_downstream",
    ]:
        if sys.modules.get(mod_name, _sentinel) is _sentinel:
            monkeypatch.setitem(sys.modules, mod_name, MagicMock())

    monkeypatch.setitem(sys.modules, "submods.croco.models.croco_downstream", fake_croco_downstream)

    ckpt_file = tmp_path / "croco_fake.pth"
    ckpt_file.write_bytes(b"")

    monkeypatch.setattr(
        "omniprobe.models.croco._get_croco_encoder_no_head_class",
        lambda: _FakeCroCoEncoder,
    )

    import torch as _torch
    monkeypatch.setattr(_torch, "load", lambda *a, **kw: {"model": {}})
    monkeypatch.setattr(
        "omniprobe.models.croco.resolve_pretrained_path", lambda *a, **kw: ckpt_file
    )
    return _FakeCroCoEncoder


# ---------------------------------------------------------------------------
# MetaCLIP fixture (MetaCLIP submodule factory + checkpoint mock)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_metaclip(monkeypatch, tmp_path):
    """Patch MetaCLIP factory and checkpoint loading."""
    fake_clip = _FakeCLIPModel(embed_dim=768, patch_size=16, num_blocks=12, img_size=224)
    monkeypatch.setattr(
        "omniprobe.models.metaclip._create_model_and_transforms",
        lambda *a, **kw: (fake_clip, None, None),
    )

    ckpt_file = tmp_path / "metaclip_fake.pt"
    ckpt_file.write_bytes(b"")

    import torch as _torch
    monkeypatch.setattr(_torch, "load", lambda *a, **kw: {})
    monkeypatch.setattr(
        "omniprobe.models.metaclip.resolve_pretrained_reference",
        lambda *a, **kw: ckpt_file,
    )
    return fake_clip


# ---------------------------------------------------------------------------
# Perception fake ViT (perception_models submodule)
# ---------------------------------------------------------------------------

class _FakePerceptionTransformer:
    def __init__(self, width, num_layers):
        self.width = width
        self.resblocks = [nn.Identity() for _ in range(num_layers)]


class _FakePerceptionViT(nn.Module):
    """Fake pe.VisionTransformer for PerceptionBackbone tests."""

    def __init__(self, embed_dim=768, patch_size=16, num_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.layers = num_layers
        self.width = embed_dim

        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.class_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.ln_pre = nn.Identity()
        self.ln_post = nn.Identity()

        self.use_cls_token = True
        self.use_abs_posemb = True
        self.use_rope2d = False

        self.transformer = _FakePerceptionTransformer(embed_dim, num_layers)

    def _sample_abs_posemb(self, grid_h, grid_w):
        return torch.zeros(1, 1 + grid_h * grid_w, self.width)

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_perception(monkeypatch, tmp_path):
    """Patch pe.VisionTransformer.from_config and return the tmp checkpoint path."""
    from omniprobe.models.vendor.perception_models.core.vision_encoder import pe as _pe

    monkeypatch.setattr(_pe.VisionTransformer, "from_config",
                        classmethod(lambda cls, *a, **kw: _FakePerceptionViT()))

    ckpt_file = tmp_path / "pe_fake.pt"
    ckpt_file.write_bytes(b"")
    return ckpt_file


# ---------------------------------------------------------------------------
# VGGT fake model (vggt submodule)
# ---------------------------------------------------------------------------

class _FakeVGGTAggPatchEmbed(nn.Module):
    """Fake DINOv2 patch embed inside the VGGT aggregator."""

    def __init__(self, embed_dim=768, patch_size=14, depth=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])

    def forward(self, images):
        B = images.shape[0]
        h = images.shape[-2] // self.patch_size
        w = images.shape[-1] // self.patch_size
        return torch.zeros(B, h * w, self.embed_dim)

    def get_intermediate_layers(self, images, n, reshape=True, **kwargs):
        B = images.shape[0]
        h = images.shape[-2] // self.patch_size
        w = images.shape[-1] // self.patch_size
        if reshape:
            feat = torch.zeros(B, self.embed_dim, h, w)
        else:
            feat = torch.zeros(B, h * w, self.embed_dim)
        count = len(n) if isinstance(n, (list, tuple)) else n
        return [feat] * count


class _FakeVGGTAggregator(nn.Module):
    def __init__(self, embed_dim=768, patch_size=14, depth=12):
        super().__init__()
        self.patch_size = patch_size
        self.depth = depth
        self.patch_embed = _FakeVGGTAggPatchEmbed(embed_dim, patch_size, depth)

        # For aggregator feature_source: frame_blocks[0].attn.proj.out_features
        class _FakeProj:
            out_features = embed_dim * 2
        class _FakeAttn:
            proj = _FakeProj()
        class _FakeFrameBlock:
            attn = _FakeAttn()
        self.frame_blocks = [_FakeFrameBlock()]

        # Normalization tensors for patch_embed source path
        self._resnet_mean = torch.zeros(1, 1, 3, 1, 1)
        self._resnet_std = torch.ones(1, 1, 3, 1, 1)


class _FakeVGGTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = _FakeVGGTAggregator()

    def load_state_dict(self, state_dict, strict=True):
        return [], []

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


@pytest.fixture
def fake_vggt(monkeypatch, tmp_path):
    """Patch VGGT class and checkpoint loading for VGGTBackbone tests."""
    monkeypatch.setattr("omniprobe.models.vggt._VGGT", _FakeVGGTModel)

    ckpt_file = tmp_path / "vggt_fake.pt"
    ckpt_file.write_bytes(b"")

    import torch as _torch
    monkeypatch.setattr(_torch, "load", lambda *a, **kw: {})
    monkeypatch.setattr(
        "omniprobe.models.vggt.resolve_pretrained_path", lambda *a, **kw: ckpt_file
    )
    return _FakeVGGTModel
