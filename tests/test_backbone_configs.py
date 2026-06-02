"""Per-config backbone tests: output mode shapes and feat_dim consistency.

For every backbone config that can be instantiated without real weights
(hub-loaded: DINO, RADIO, C-RADIO, DUNE, VJEPA2), this file:
  1. Instantiates each named config with a tiny fake model.
  2. Verifies feat_dim matches the config contract.
  3. Runs a forward pass for EVERY supported output mode.
  4. Asserts the output tensor shape is consistent with feat_dim.

Backbones that require real checkpoints (CroCo, IJEPA, CLIP, MAE, …)
are NOT instantiated here — their contract metadata is already checked
by test_backbone_contracts.py.
"""

import pytest
import torch

from tests.conftest import _FakeDinoViT, _FakeRADIOModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_output_shape(output, feat_dim, output_mode):
    """Assert that a backbone forward output has the right shape for the mode."""
    # Unwrap single-layer list if necessary
    if isinstance(output, (list, tuple)):
        output = output[0]

    if output_mode == "dense":
        assert output.dim() == 4, f"{output_mode}: expected (B,C,H,W), got {output.shape}"
        assert output.shape[1] == feat_dim, (
            f"{output_mode}: channel dim {output.shape[1]} != feat_dim {feat_dim}"
        )
    else:  # cls, gap, map
        assert output.dim() == 2, f"{output_mode}: expected (B,C), got {output.shape}"
        assert output.shape[1] == feat_dim, (
            f"{output_mode}: channel dim {output.shape[1]} != feat_dim {feat_dim}"
        )


# ---------------------------------------------------------------------------
# DINO / DINOv2 configs
# (embed_dim and patch_size must match what DINO.__init__ will read back)
# ---------------------------------------------------------------------------
#   (config_name,  dino_name,  model_name,    embed_dim, patch_size)
_DINO_CASES = [
    ("dino_b16",       "dino",   "vitb16",     768, 16),
    ("dino_b8",        "dino",   "vitb8",      768,  8),
    ("dino_s16",       "dino",   "vits16",     384, 16),
    ("dinov2_b14",     "dinov2", "vitb14",     768, 14),
    ("dinov2_s14",     "dinov2", "vits14",     384, 14),
    ("dinov2_l14",     "dinov2", "vitl14",    1024, 14),
    ("dinov2_g14",     "dinov2", "vitg14",    1536, 14),
    ("dinov2_b14_reg", "dinov2", "vitb14_reg", 768, 14),
]

_DINO_OUTPUTS = ["dense", "cls", "gap"]


@pytest.mark.parametrize("cfg_name,dino_name,model_name,embed_dim,patch_size", _DINO_CASES)
@pytest.mark.parametrize("output_mode", _DINO_OUTPUTS)
def test_dino_config_output_shape(
    cfg_name, dino_name, model_name, embed_dim, patch_size, output_mode, monkeypatch
):
    from omniprobe.models.dino import DINO

    fake = _FakeDinoViT(embed_dim=embed_dim, patch_size=patch_size)
    fake.patch_embed.proj.kernel_size = (patch_size, patch_size)
    monkeypatch.setattr("torch.hub.load", lambda *a, **kw: fake)

    model = DINO(dino_name=dino_name, model_name=model_name, output=output_mode)

    expected_feat_dim = embed_dim
    assert model.feat_dim == expected_feat_dim, (
        f"{cfg_name}/{output_mode}: feat_dim={model.feat_dim}, expected {expected_feat_dim}"
    )

    x = torch.zeros(1, 3, patch_size * 16, patch_size * 16)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


@pytest.mark.parametrize("cfg_name,dino_name,model_name,embed_dim,patch_size", _DINO_CASES)
def test_dino_config_multilayer_feat_dim(
    cfg_name, dino_name, model_name, embed_dim, patch_size, monkeypatch
):
    from omniprobe.models.dino import DINO

    fake = _FakeDinoViT(embed_dim=embed_dim, patch_size=patch_size)
    fake.patch_embed.proj.kernel_size = (patch_size, patch_size)
    monkeypatch.setattr("torch.hub.load", lambda *a, **kw: fake)

    model = DINO(dino_name=dino_name, model_name=model_name, output="dense", return_multilayer=True)

    assert isinstance(model.feat_dim, list), f"{cfg_name}: expected list feat_dim"
    assert len(model.feat_dim) == 4
    assert all(d == embed_dim for d in model.feat_dim), (
        f"{cfg_name}: multilayer dims {model.feat_dim}, expected all {embed_dim}"
    )

    x = torch.zeros(1, 3, patch_size * 16, patch_size * 16)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, embed_dim, "dense")


# ---------------------------------------------------------------------------
# C-RADIO configs
# ---------------------------------------------------------------------------
_CRADIO_CASES = [
    ("c_radio_3_b",      "CRADIOv3Backbone", "c-radio_v3-b"),
    ("c_radio_3_l",      "CRADIOv3Backbone", "c-radio_v3-l"),
    ("c_radio_3_h",      "CRADIOv3Backbone", "c-radio_v3-h"),
    ("c_radio_3_g",      "CRADIOv3Backbone", "c-radio_v3-g"),
    ("c_radio_4_h",      "CRADIOv4Backbone", "c-radio_v4-h"),
    ("c_radio_4_so400m", "CRADIOv4Backbone", "c-radio_v4-so400m"),
]

_CRADIO_OUTPUTS = ["dense", "gap", "cls"]


@pytest.mark.parametrize("cfg_name,cls_name,version", _CRADIO_CASES)
@pytest.mark.parametrize("output_mode", _CRADIO_OUTPUTS)
def test_c_radio_config_output_shape(cfg_name, cls_name, version, output_mode, monkeypatch):
    import omniprobe.models.c_radio as c_radio_mod

    fake = _FakeRADIOModel(embed_dim=1024, patch_size=14)
    monkeypatch.setattr("torch.hub.load", lambda *a, **kw: fake)

    cls = getattr(c_radio_mod, cls_name)
    model = cls(version=version, output=output_mode)

    assert model.feat_dim == 1024

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


# ---------------------------------------------------------------------------
# DUNE configs
# ---------------------------------------------------------------------------
#   (config_name, arch, embed_dim, patch_size)
_DUNE_CASES = [
    ("dune_vits14_448",       "vits14_448",       384, 14),
    ("dune_vitb14_336",       "vitb14_336",       768, 14),
    ("dune_vitb14",           "vitb14_448",       768, 14),
    ("dune_vitb14_448_paper", "vitb14_448_paper", 768, 14),
]

_DUNE_OUTPUTS = ["dense", "gap", "cls"]


@pytest.mark.parametrize("cfg_name,arch,embed_dim,patch_size", _DUNE_CASES)
@pytest.mark.parametrize("output_mode", _DUNE_OUTPUTS)
def test_dune_config_output_shape(cfg_name, arch, embed_dim, patch_size, output_mode, monkeypatch):
    from omniprobe.models.dune import DUNE

    fake = _FakeDinoViT(embed_dim=embed_dim, patch_size=patch_size)
    monkeypatch.setattr("torch.hub.load", lambda *a, **kw: fake)

    model = DUNE(arch=arch, output=output_mode)

    expected_feat_dim = embed_dim
    assert model.feat_dim == expected_feat_dim, (
        f"{cfg_name}/{output_mode}: feat_dim={model.feat_dim}, expected {expected_feat_dim}"
    )

    x = torch.zeros(1, 3, patch_size * 16, patch_size * 16)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


# ---------------------------------------------------------------------------
# VJEPA2 configs
# ---------------------------------------------------------------------------
#   (config_name, model_type, embed_dim)
_VJEPA2_CASES = [
    ("vjepa2_1_base",  "vjepa2_1_vit_base_384",  768),
    ("vjepa2_1_large", "vjepa2_1_vit_large_384", 1024),
    ("vjepa2_large",   "vjepa2_vit_large",       1024),
]

_VJEPA2_OUTPUTS = ["dense", "gap"]


@pytest.mark.parametrize("cfg_name,model_type,embed_dim", _VJEPA2_CASES)
@pytest.mark.parametrize("output_mode", _VJEPA2_OUTPUTS)
def test_vjepa2_config_output_shape(cfg_name, model_type, embed_dim, output_mode, monkeypatch):
    from omniprobe.models.vjepa2 import VJEPA2Backbone

    fake = _FakeDinoViT(embed_dim=embed_dim, patch_size=16)
    fake.num_features = embed_dim
    monkeypatch.setattr("torch.hub.load", lambda *a, **kw: fake)

    model = VJEPA2Backbone(model_type=model_type, output=output_mode, pretrained=False)

    assert model.feat_dim == embed_dim, (
        f"{cfg_name}/{output_mode}: feat_dim={model.feat_dim}, expected {embed_dim}"
    )

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)
