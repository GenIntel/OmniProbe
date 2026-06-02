from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import omniprobe.models.contracts as contracts
from omniprobe.models.contracts import (
    get_backbone_contract,
    instantiate_backbone_for_output,
    validate_multilayer_feat_dim,
)


def test_backbone_contract_from_cfg_metadata():
    cfg = OmegaConf.create(
        {
            "_target_": "tests.fakes.TinyDenseBackbone",
            "supported_outputs": ["dense", "gap"],
            "default_global_output": "gap",
            "supports_multilayer": True,
            "supports_layer_selection": True,
            "image_mean": "imagenet",
        }
    )
    contract = get_backbone_contract(cfg)
    assert contract.supported_outputs == ("dense", "gap")
    assert contract.resolve_global_output() == "gap"
    assert contract.input_normalization == "imagenet"


def test_single_and_multilayer_feat_dim_consistency():
    cfg = OmegaConf.create(
        {
            "_target_": "tests.fakes.TinyDenseBackbone",
            "supported_outputs": ["dense", "gap"],
            "default_global_output": "gap",
            "supports_multilayer": True,
            "supports_layer_selection": True,
            "image_mean": "imagenet",
        }
    )
    single_model, _ = instantiate_backbone_for_output(
        cfg,
        output_name="dense",
        return_multilayer=False,
        device=torch.device("cpu"),
    )
    multi_model, _ = instantiate_backbone_for_output(
        cfg,
        output_name="dense",
        return_multilayer=True,
        device=torch.device("cpu"),
    )
    assert isinstance(single_model.feat_dim, int)
    assert isinstance(multi_model.feat_dim, list)
    validate_multilayer_feat_dim(single_model)
    validate_multilayer_feat_dim(multi_model)


def test_backbone_configs_have_contract_or_inline_metadata():
    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "backbone"
    for path in sorted(cfg_dir.glob("*.yaml")):
        cfg = OmegaConf.load(path)
        contract = get_backbone_contract(cfg)
        assert len(contract.supported_outputs) > 0, path.name


def test_c_radio_contract_disables_layer_selection():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs" / "backbone" / "c_radio_3_b.yaml")
    contract = get_backbone_contract(cfg)
    assert contract.supports_layer_selection is False


@pytest.mark.parametrize(
    ("backbone_name", "expected_target"),
    [
        ("c_radio_3_b", "omniprobe.models.c_radio.CRADIOv3Backbone"),
        ("c_radio_4_h", "omniprobe.models.c_radio.CRADIOv4Backbone"),
        ("clip_b16", "omniprobe.models.clip.CLIP"),
        ("dino_b16", "omniprobe.models.dino.DINO"),
        ("dinov2_b14", "omniprobe.models.dino.DINO"),
    ],
)
def test_instantiate_backbone_for_output_accepts_struct_backbone_configs(
    monkeypatch,
    backbone_name,
    expected_target,
):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs" / "backbone" / f"{backbone_name}.yaml")
    seen = {}

    class DummyModel:
        feat_dim = 8

        def to(self, device):
            seen["device"] = str(device)
            return self

        def eval(self):
            seen["eval"] = True
            return self

    def fake_instantiate(config, **kwargs):
        seen["config"] = OmegaConf.to_container(config, resolve=True)
        seen["kwargs"] = kwargs
        return DummyModel()

    monkeypatch.setattr(contracts, "instantiate", fake_instantiate)
    model, contract = instantiate_backbone_for_output(
        cfg,
        output_name="dense",
        return_multilayer=False,
        device=torch.device("cpu"),
    )
    assert contract.target == expected_target
    assert seen["config"]["_target_"] == expected_target
    assert "image_mean" not in seen["config"]
    assert "supported_outputs" not in seen["config"]
    assert seen["kwargs"]["output"] == "dense"
    assert seen["kwargs"]["return_multilayer"] is False
    assert seen["device"] == "cpu"
    assert seen["eval"] is True
    assert model.feat_dim == 8


@pytest.mark.parametrize(
    ("backbone_name", "expects_layer"),
    [
        ("c_radio_3_b", False),
        ("dino_b16", True),
    ],
)
def test_instantiate_backbone_for_output_only_passes_supported_kwargs(
    monkeypatch,
    backbone_name,
    expects_layer,
):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs" / "backbone" / f"{backbone_name}.yaml")
    seen = {}

    class DummyModel:
        feat_dim = 8

        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_instantiate(config, **kwargs):
        seen["kwargs"] = kwargs
        return DummyModel()

    monkeypatch.setattr(contracts, "instantiate", fake_instantiate)
    instantiate_backbone_for_output(
        cfg,
        output_name="dense",
        return_multilayer=False,
        device=torch.device("cpu"),
        layer=7,
    )
    assert ("layer" in seen["kwargs"]) is expects_layer
