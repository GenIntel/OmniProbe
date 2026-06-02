"""Tests for backbone model loading.

Two layers of coverage:
1. Static analysis — every remote torch.hub.load call must pass trust_repo=True
   so that models load from the local cache on compute nodes without internet.
2. Instantiation + forward — DINO/DINOv2 backbones are instantiated with a fake
   ViT (no weights, no GPU) and their output shapes are verified.
"""

import ast
import importlib
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Static analysis: trust_repo enforcement
# ---------------------------------------------------------------------------

def _iter_hub_load_calls(source: str):
    """Yield AST Call nodes that look like torch.hub.load(...)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "hub"
        ):
            continue
        yield node


def _kw_literal(node: ast.Call, name: str):
    """Return the literal value of keyword argument ``name``, or None."""
    for kw in node.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def test_all_remote_hub_loads_have_trust_repo():
    """Every torch.hub.load that contacts a remote repo must pass trust_repo=True.

    This prevents the GitHub fork-validation step from failing on compute nodes
    that have no internet access, even when the model is already cached locally.
    """
    models_dir = Path(__file__).resolve().parents[1] / "omniprobe" / "models"
    violations = []
    for path in sorted(models_dir.glob("*.py")):
        source = path.read_text()
        if "torch.hub.load" not in source:
            continue
        for node in _iter_hub_load_calls(source):
            has_local_source = _kw_literal(node, "source") == "local"
            has_trust_repo = _kw_literal(node, "trust_repo") is True
            if not has_local_source and not has_trust_repo:
                violations.append(f"{path.name}:{node.lineno}")

    assert not violations, (
        "These torch.hub.load calls are missing trust_repo=True and will fail "
        "on compute nodes without internet access:\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Instantiation tests: DINO / DINOv2 (use fake_dino_hub fixture from conftest)
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


def test_dino_backbone_multilayer_feat_dim(fake_dino_hub):
    from omniprobe.models.dino import DINO

    model = DINO(dino_name="dinov2", model_name="vitb14", output="dense", return_multilayer=True)
    assert isinstance(model.feat_dim, list)
    assert len(model.feat_dim) == 4
    assert all(d == 768 for d in model.feat_dim)


def test_dino_backbone_forward_dense(fake_dino_hub):
    from omniprobe.models.dino import DINO

    model = DINO(dino_name="dinov2", model_name="vitb14", output="dense")
    x = torch.zeros(1, 3, 224, 224)
    out = model(x)

    assert out.dim() == 4, f"Expected (B,C,H,W), got {out.shape}"
    B, C, H, W = out.shape
    assert B == 1
    assert C == 768


def test_dino_backbone_forward_gap(fake_dino_hub):
    from omniprobe.models.dino import DINO

    model = DINO(dino_name="dinov2", model_name="vitb14", output="gap")
    x = torch.zeros(1, 3, 224, 224)
    out = model(x)

    assert out.dim() == 2, f"Expected (B,C), got {out.shape}"
    assert out.shape == (1, 768)
