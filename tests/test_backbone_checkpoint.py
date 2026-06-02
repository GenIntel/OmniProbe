"""Checkpoint-backed backbone tests: output mode shapes and feat_dim consistency.

For backbones that require real checkpoints (CLIP, MAE, DeiT, SigLIP, iBOT,
SAM, DINOv3, PIXIO, IJEPA), this file:
  1. Monkeypatches the model loader (open_clip, transformers, timm, local vit
     factory, SAM registry, etc.) to return a tiny fake model.
  2. Instantiates the backbone wrapper.
  3. Asserts feat_dim matches the expected value.
  4. Runs a forward pass and asserts the output shape for every supported mode.
  5. Asserts multilayer feat_dim consistency where supported.

No real weights, GPU, or network access required.
"""

import pytest
import torch

from tests.conftest import (
    _FakeDinoViT,
    _FakeRADIOModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_output_shape(output, feat_dim, output_mode):
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output_mode == "dense":
        assert output.dim() == 4, f"{output_mode}: expected (B,C,H,W), got {output.shape}"
        assert output.shape[1] == feat_dim, (
            f"{output_mode}: channel dim {output.shape[1]} != feat_dim {feat_dim}"
        )
    else:
        assert output.dim() == 2, f"{output_mode}: expected (B,C), got {output.shape}"
        assert output.shape[1] == feat_dim, (
            f"{output_mode}: channel dim {output.shape[1]} != feat_dim {feat_dim}"
        )


# ---------------------------------------------------------------------------
# CLIP (open_clip)
# ---------------------------------------------------------------------------

_CLIP_OUTPUTS = ["dense", "cls", "gap"]


@pytest.mark.parametrize("output_mode", _CLIP_OUTPUTS)
def test_clip_output_shape(output_mode, fake_openclip):
    from omniprobe.models.clip import CLIP

    model = CLIP(arch="ViT-B-16", checkpoint="openai", output=output_mode)

    assert model.feat_dim == 768

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_clip_multilayer_feat_dim(fake_openclip):
    from omniprobe.models.clip import CLIP

    model = CLIP(arch="ViT-B-16", checkpoint="openai", output="dense",
                 return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 768 for d in model.feat_dim)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 768, "dense")


# ---------------------------------------------------------------------------
# MAE (HuggingFace ViTMAE)
# ---------------------------------------------------------------------------

_MAE_OUTPUTS = ["dense", "cls", "gap"]


@pytest.mark.parametrize("output_mode", _MAE_OUTPUTS)
def test_mae_output_shape(output_mode, fake_hf_mae):
    from omniprobe.models.mae import MAE

    model = MAE(checkpoint="facebook/vit-mae-base", output=output_mode)

    assert model.feat_dim == 768

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_mae_multilayer_feat_dim(fake_hf_mae):
    from omniprobe.models.mae import MAE

    model = MAE(checkpoint="facebook/vit-mae-base", output="dense",
                return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 768 for d in model.feat_dim)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 768, "dense")


# ---------------------------------------------------------------------------
# DeiT (deit_utils factory)
# ---------------------------------------------------------------------------

_DEIT_OUTPUTS = ["dense", "cls", "gap"]


@pytest.mark.parametrize("output_mode", _DEIT_OUTPUTS)
def test_deit_output_shape(output_mode, fake_deit):
    from omniprobe.models.deit import DeIT

    model = DeIT(model_size="base", img_size=384, patch_size=16, output=output_mode)

    assert model.feat_dim == 768

    x = torch.zeros(1, 3, 384, 384)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_deit_multilayer_feat_dim(fake_deit):
    from omniprobe.models.deit import DeIT

    model = DeIT(model_size="base", img_size=384, patch_size=16, output="dense",
                 return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 768 for d in model.feat_dim)

    x = torch.zeros(1, 3, 384, 384)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 768, "dense")


# ---------------------------------------------------------------------------
# SigLIP (timm)
# ---------------------------------------------------------------------------

_SIGLIP_OUTPUTS = ["dense", "gap"]


@pytest.mark.parametrize("output_mode", _SIGLIP_OUTPUTS)
def test_siglip_output_shape(output_mode, fake_timm_siglip):
    from omniprobe.models.siglip import SigLIP

    model = SigLIP(checkpoint="vit_large_patch16_siglip_384", output=output_mode,
                   pretrained=False)

    assert model.feat_dim == 1024

    x = torch.zeros(1, 3, 384, 384)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_siglip_multilayer_feat_dim(fake_timm_siglip):
    from omniprobe.models.siglip import SigLIP

    model = SigLIP(checkpoint="vit_large_patch16_siglip_384", output="dense",
                   pretrained=False, return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 1024 for d in model.feat_dim)

    x = torch.zeros(1, 3, 384, 384)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 1024, "dense")


# ---------------------------------------------------------------------------
# iBOT (ibot_transformers + torch.load)
# ---------------------------------------------------------------------------

_IBOT_OUTPUTS = ["dense", "cls", "gap"]


@pytest.mark.parametrize("output_mode", _IBOT_OUTPUTS)
def test_ibot_output_shape(output_mode, fake_ibot):
    from omniprobe.models.ibot import iBOT

    model = iBOT(model_type="base", output=output_mode)

    assert model.feat_dim == 768

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_ibot_multilayer_feat_dim(fake_ibot):
    from omniprobe.models.ibot import iBOT

    model = iBOT(model_type="base", output="dense", return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 768 for d in model.feat_dim)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 768, "dense")


# ---------------------------------------------------------------------------
# SAM (segment_anything registry)
# ---------------------------------------------------------------------------

_SAM_OUTPUTS = ["dense", "gap"]


@pytest.mark.parametrize("output_mode", _SAM_OUTPUTS)
def test_sam_output_shape(output_mode, fake_sam):
    from omniprobe.models.sam import SAM

    model = SAM(arch="vit_b", output=output_mode)

    assert model.feat_dim == 256

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_sam_multilayer_feat_dim(fake_sam):
    from omniprobe.models.sam import SAM

    model = SAM(arch="vit_b", output="dense", return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 256 for d in model.feat_dim)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 256, "dense")


# ---------------------------------------------------------------------------
# DINOv3 (dinov3.hub.backbones factory)
# ---------------------------------------------------------------------------

_DINOV3_OUTPUTS = ["dense", "cls", "gap"]


@pytest.mark.parametrize("output_mode", _DINOV3_OUTPUTS)
def test_dinov3_output_shape(output_mode, fake_dinov3):
    from omniprobe.models.dinov3 import DinoV3

    model = DinoV3(arch="vitb16", output=output_mode, pretrained=False)

    assert model.feat_dim == 768

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_dinov3_multilayer_feat_dim(fake_dinov3):
    from omniprobe.models.dinov3 import DinoV3

    model = DinoV3(arch="vitb16", output="dense", pretrained=False,
                   return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 768 for d in model.feat_dim)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 768, "dense")


# ---------------------------------------------------------------------------
# PIXIO (pixio factory functions)
# ---------------------------------------------------------------------------

_PIXIO_OUTPUTS = ["dense", "cls", "gap"]


@pytest.mark.parametrize("output_mode", _PIXIO_OUTPUTS)
def test_pixio_output_shape(output_mode, fake_pixio):
    from omniprobe.models.pixio import PIXIO

    model = PIXIO(arch="vitb16", output=output_mode)

    assert model.feat_dim == 768

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_pixio_multilayer_feat_dim(fake_pixio):
    from omniprobe.models.pixio import PIXIO

    model = PIXIO(arch="vitb16", output="dense", return_multilayer=True)

    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4
    assert all(d == 768 for d in model.feat_dim)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, 768, "dense")


# ---------------------------------------------------------------------------
# IJEPA (src.models.vision_transformer.vit_huge)
# ---------------------------------------------------------------------------

_IJEPA_OUTPUTS = ["dense", "gap"]


@pytest.mark.parametrize("output_mode", _IJEPA_OUTPUTS)
def test_ijepa_output_shape(output_mode, fake_ijepa):
    from omniprobe.models.ijepa import IJEPA

    model = IJEPA(model_type="h.16.test", output=output_mode)

    assert model.feat_dim == 1280

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_ijepa_multilayer_feat_dim(fake_ijepa):
    from omniprobe.models.ijepa import IJEPA

    model = IJEPA(model_type="h.16.test", output="dense", return_multilayer=True)

    assert isinstance(model.feat_dim, list)
    assert all(d == 1280 for d in model.feat_dim)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list)
    for layer_out in out:
        _check_output_shape(layer_out, 1280, "dense")


# ---------------------------------------------------------------------------
# Multilayer forward tests for hub-loaded backbones
# (feat_dim tested elsewhere; these verify the forward pass returns correct shapes)
# ---------------------------------------------------------------------------

def test_c_radio_multilayer_forward(fake_radio_hub):
    from omniprobe.models.c_radio import CRADIOv4Backbone

    model = CRADIOv4Backbone(version="c-radio_v4-h", output="dense",
                              return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


def test_dune_multilayer_forward(fake_dino_hub):
    from omniprobe.models.dune import DUNE

    model = DUNE(arch="vitb14_448", output="dense", return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


def test_vjepa2_multilayer_forward(fake_dino_hub):
    from omniprobe.models.vjepa2 import VJEPA2Backbone

    model = VJEPA2Backbone(model_type="vjepa2_1_vit_base_384", output="dense",
                           pretrained=False, return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


def test_dino_reg_multilayer_forward(fake_dino_hub):
    from omniprobe.models.dino_reg import DINO_REG

    model = DINO_REG(dino_name="dinov2", model_name="vitb14", output="dense",
                     return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


# ===========================================================================
# ConvNext
# ===========================================================================

@pytest.mark.parametrize("output_mode", ["dense", "gap"])
def test_convnext_output_shape(output_mode, fake_convnext):
    from omniprobe.models.convnext import ConvNext

    model = ConvNext(
        arch="convnext_base_w",
        checkpoint="laion2b_s13b_b82k",
        output=output_mode,
    )
    assert isinstance(model.feat_dim, int)
    assert "dense" in model.supported_outputs if hasattr(model, "supported_outputs") else True

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_convnext_multilayer_feat_dim(fake_convnext):
    from omniprobe.models.convnext import ConvNext

    model = ConvNext(
        arch="convnext_base_w",
        checkpoint="laion2b_s13b_b82k",
        output="dense",
        return_multilayer=True,
    )
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


# ===========================================================================
# MiDaS (midas=True path: torch.hub.load DPT_Large -> midas_forward)
# ===========================================================================

@pytest.mark.parametrize("output_mode", ["dense", "gap"])
def test_midas_output_shape(output_mode, fake_midas_hub):
    from omniprobe.models.midas_final import make_beit_backbone

    model = make_beit_backbone(output=output_mode, midas=True)
    assert isinstance(model.feat_dim, int)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_midas_multilayer_feat_dim(fake_midas_hub):
    from omniprobe.models.midas_final import make_beit_backbone

    model = make_beit_backbone(output="dense", midas=True, return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


# ===========================================================================
# DIY-SC (DINOv2 + AggregationNetwork)
# ===========================================================================

@pytest.mark.parametrize("output_mode", ["dense", "gap"])
def test_diy_sc_output_shape(output_mode, fake_diy_sc):
    from omniprobe.models.dino_diy_sc import DINO

    model = DINO(output=output_mode)
    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_diy_sc_multilayer_feat_dim(fake_diy_sc):
    from omniprobe.models.dino_diy_sc import DINO

    model = DINO(output="dense", return_multilayer=True)
    assert isinstance(model.feat_dim, list)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list)
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


# ===========================================================================
# CroCo (requires croco submodule)
# ===========================================================================

@pytest.mark.parametrize("output_mode", ["dense", "gap"])
def test_croco_output_shape(output_mode, fake_croco):
    from omniprobe.models.croco import CroCoBackbone

    model = CroCoBackbone(output=output_mode, image_size=224)
    assert isinstance(model.feat_dim, int)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_croco_multilayer_feat_dim(fake_croco):
    from omniprobe.models.croco import CroCoBackbone

    model = CroCoBackbone(output="dense", image_size=224, return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


# ===========================================================================
# MetaCLIP (requires MetaCLIP submodule)
# ===========================================================================

@pytest.mark.parametrize("output_mode", ["dense", "cls", "gap"])
def test_metaclip_output_shape(output_mode, fake_metaclip):
    from omniprobe.models.metaclip import MetaCLIP

    # Use vitb16 so fake embed_dim=768 matches _ARCH_CONFIGS["vitb16"]
    model = MetaCLIP(arch="vitb16", checkpoint="fake_ckpt.pt", output=output_mode)
    expected_feat = model.feat_dim

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, expected_feat, output_mode)


def test_metaclip_multilayer_feat_dim(fake_metaclip):
    from omniprobe.models.metaclip import MetaCLIP

    model = MetaCLIP(arch="vitb16", checkpoint="fake_ckpt.pt", output="dense",
                     return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list) and len(out) == 4
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


# ===========================================================================
# Perception Encoder (requires perception_models submodule)
# ===========================================================================

@pytest.mark.parametrize("output_mode", ["dense", "gap"])
def test_perception_output_shape(output_mode, fake_perception):
    from omniprobe.models.perception import PerceptionBackbone

    model = PerceptionBackbone(
        model_type="pe_core_g14_448",
        checkpoint_path=str(fake_perception),
        output=output_mode,
    )
    assert isinstance(model.feat_dim, int)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, output_mode)


def test_perception_multilayer_feat_dim(fake_perception):
    from omniprobe.models.perception import PerceptionBackbone

    model = PerceptionBackbone(
        model_type="pe_core_g14_448",
        checkpoint_path=str(fake_perception),
        output="dense",
        return_multilayer=True,
    )
    assert isinstance(model.feat_dim, list)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list)
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")


# ===========================================================================
# VGGT (requires vggt submodule)
# ===========================================================================

def test_vggt_output_shape(fake_vggt):
    from omniprobe.models.vggt import VGGTBackbone

    model = VGGTBackbone(feature_source="patch_embed", output="dense")
    assert isinstance(model.feat_dim, int)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    _check_output_shape(out, model.feat_dim, "dense")


def test_vggt_multilayer_feat_dim(fake_vggt):
    from omniprobe.models.vggt import VGGTBackbone

    model = VGGTBackbone(
        feature_source="patch_embed", output="dense", return_multilayer=True
    )
    assert isinstance(model.feat_dim, list)

    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert isinstance(out, list)
    for layer_out in out:
        _check_output_shape(layer_out, model.feat_dim[0], "dense")
