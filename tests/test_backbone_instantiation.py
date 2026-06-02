"""Instantiation and import tests for all backbone modules.

Coverage layers:
1. Module import smoke — every omniprobe/models/*.py must be importable.
2. Hub-loaded backbones — C-RADIO, DUNE, DINO_REG, VJEPA2 are
   instantiated against a tiny fake model (no weights, no GPU).
3. Vendored code importability — the key classes used by CroCo, IJEPA,
   VGGT, Perception, PIXIO, and MetaCLIP can be imported from the
   vendored packages under omniprobe/models/vendor/.
"""

import importlib

import pytest
import torch

# ---------------------------------------------------------------------------
# 1. Module import smoke tests
# ---------------------------------------------------------------------------

_BACKBONE_MODULES = [
    "omniprobe.models.clip",
    "omniprobe.models.convnext",
    "omniprobe.models.croco",
    "omniprobe.models.c_radio",
    "omniprobe.models.deit",
    "omniprobe.models.dino",
    "omniprobe.models.dino_diy_sc",
    "omniprobe.models.dino_reg",
    "omniprobe.models.dinov3",
    "omniprobe.models.dune",
    "omniprobe.models.ibot",
    "omniprobe.models.ijepa",
    "omniprobe.models.mae",
    "omniprobe.models.metaclip",
    "omniprobe.models.midas_final",
    "omniprobe.models.perception",
    "omniprobe.models.pixio",
    "omniprobe.models.sam",
    "omniprobe.models.siglip",
    "omniprobe.models.vggt",
    "omniprobe.models.vjepa2",
]

# These modules require optional heavyweight dependencies. Each is paired with the
# top-level package that gates it; the test skips only when that package is absent,
# and otherwise actually imports the module.
_OPTIONAL_BACKBONE_MODULES = [
    ("omniprobe.models.lvlm_visual", "transformers"),     # VL bindings
    ("omniprobe.models.stablediffusion", "diffusers"),    # latent diffusion
]


@pytest.mark.parametrize("module_name", _BACKBONE_MODULES)
def test_backbone_module_is_importable(module_name):
    """Every backbone module must import without error (catches missing deps,
    syntax errors, and bad top-level code)."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name,dependency", _OPTIONAL_BACKBONE_MODULES)
def test_optional_backbone_module_importable(module_name, dependency):
    """Optional-dependency backbone modules: skip only if the gating dependency is
    absent; otherwise the module must import without error."""
    pytest.importorskip(dependency)
    importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# 2. Hub-loaded backbone instantiation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("dino_name", "model_name", "expected_feat_dim"),
    [
        ("dino", "vitb16", 768),
        ("dinov2", "vitb14", 768),
        ("dinov2", "vitl14", 1024),
    ],
)
def test_dino_backbone_instantiates(dino_name, model_name, expected_feat_dim, fake_dino_hub):
    from omniprobe.models.dino import DINO

    fake_dino_hub.embed_dim = expected_feat_dim
    model = DINO(dino_name=dino_name, model_name=model_name, output="dense")
    assert model.feat_dim == expected_feat_dim
    assert model.checkpoint_name.startswith(dino_name)
    assert isinstance(model.patch_size, int)


def test_dino_forward_dense(fake_dino_hub):
    from omniprobe.models.dino import DINO

    model = DINO(dino_name="dinov2", model_name="vitb14", output="dense")
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.dim() == 4
    assert out.shape[:2] == (1, 768)


def test_dino_forward_gap(fake_dino_hub):
    from omniprobe.models.dino import DINO

    model = DINO(dino_name="dinov2", model_name="vitb14", output="gap")
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.dim() == 2
    assert out.shape == (1, 768)


def test_dino_multilayer_feat_dim(fake_dino_hub):
    from omniprobe.models.dino import DINO

    model = DINO(dino_name="dinov2", model_name="vitb14", output="dense", return_multilayer=True)
    assert isinstance(model.feat_dim, list) and len(model.feat_dim) == 4


# --- DINO_REG ---

@pytest.mark.parametrize(
    ("dino_name", "model_name", "expected_feat_dim"),
    [
        ("dinov2", "vitb14", 768),
        ("dinov2", "vitl14", 1024),
    ],
)
def test_dino_reg_backbone_instantiates(dino_name, model_name, expected_feat_dim, fake_dino_hub):
    from omniprobe.models.dino_reg import DINO_REG

    fake_dino_hub.embed_dim = expected_feat_dim
    model = DINO_REG(dino_name=dino_name, model_name=model_name, output="dense")
    assert model.feat_dim == expected_feat_dim
    assert isinstance(model.patch_size, int)


def test_dino_reg_forward_dense(fake_dino_hub):
    from omniprobe.models.dino_reg import DINO_REG

    model = DINO_REG(dino_name="dinov2", model_name="vitb14", output="dense")
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.dim() == 4
    assert out.shape[1] == 768


# --- C-RADIO ---

@pytest.mark.parametrize("version", ["c-radio_v4-h", "c-radio_v3-b"])
def test_c_radio_backbone_instantiates(version, fake_radio_hub):
    from omniprobe.models.c_radio import CRADIOv4Backbone, CRADIOv3Backbone

    cls = CRADIOv3Backbone if "v3" in version else CRADIOv4Backbone
    model = cls(version=version, output="dense")
    assert isinstance(model.feat_dim, int)
    assert isinstance(model.patch_size, int)
    assert model.checkpoint_name == version


def test_c_radio_forward_dense(fake_radio_hub):
    from omniprobe.models.c_radio import CRADIOv4Backbone

    model = CRADIOv4Backbone(version="c-radio_v4-h", output="dense")
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.dim() == 4


# --- DUNE ---

def test_dune_backbone_instantiates(fake_dino_hub):
    from omniprobe.models.dune import DUNE

    model = DUNE(arch="vitb14_448_paper", output="dense")
    assert isinstance(model.feat_dim, int)
    assert model.patch_size == 14
    assert "dune" in model.checkpoint_name


def test_dune_multilayer_feat_dim(fake_dino_hub):
    from omniprobe.models.dune import DUNE

    model = DUNE(arch="vitb14_448_paper", output="dense", return_multilayer=True)
    assert isinstance(model.feat_dim, list)


def test_dune_forward_dense(fake_dino_hub):
    from omniprobe.models.dune import DUNE

    model = DUNE(arch="vitb14_448_paper", output="dense")
    out = model(torch.zeros(1, 3, 448, 448))
    assert out.dim() == 4
    assert out.shape[0] == 1


# --- VJEPA2 ---

def test_vjepa2_backbone_instantiates(fake_dino_hub):
    from omniprobe.models.vjepa2 import VJEPA2Backbone

    model = VJEPA2Backbone(
        model_type="vjepa2_1_vit_base_384",
        output="dense",
        pretrained=False,
    )
    assert isinstance(model.feat_dim, int)
    assert isinstance(model.patch_size, int)


def test_vjepa2_forward_dense(fake_dino_hub):
    from omniprobe.models.vjepa2 import VJEPA2Backbone

    model = VJEPA2Backbone(
        model_type="vjepa2_1_vit_base_384",
        output="dense",
        pretrained=False,
    )
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.dim() == 4


# ---------------------------------------------------------------------------
# 3. Vendored code importability tests
#
# These verify that each vendored package under omniprobe/models/vendor/ exposes
# the class/function the backbone wrapper imports.  No checkpoint required.
# ---------------------------------------------------------------------------

def test_croco_submodule_importable():
    from omniprobe.models.vendor.croco.models.croco import CroCoNet  # noqa: F401
    from omniprobe.models.vendor.croco.models.croco_downstream import croco_args_from_ckpt  # noqa: F401


def test_ijepa_submodule_importable():
    from omniprobe.models.vendor.ijepa.src.models import vision_transformer as vit  # noqa: F401
    assert hasattr(vit, "vit_huge")


def test_vggt_submodule_importable():
    import omniprobe.models.vggt  # noqa: F401 — triggers sys.path insert for vendored vggt
    from vggt.models.vggt import VGGT  # noqa: F401


def test_perception_submodule_importable():
    from omniprobe.models.vendor.perception_models.core.vision_encoder.pe import VisionTransformer  # noqa: F401


def test_pixio_submodule_importable():
    from omniprobe.models.vendor.pixio.pixio import pixio_vitb16  # noqa: F401


def test_metaclip_submodule_importable():
    from omniprobe.models.vendor.metaclip.src.mini_clip import factory  # noqa: F401


def test_dinov3_hub_model_list():
    """Verify the DINOv3 VARIANTS dict is self-consistent (no hub call needed)."""
    from omniprobe.models.dinov3 import DinoV3

    assert "vitb16" in DinoV3.VARIANTS
    assert all(
        "hub_fn" in v and "feat_dim" in v
        for v in DinoV3.VARIANTS.values()
    )
