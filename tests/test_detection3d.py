import pytest
import torch
from omegaconf import OmegaConf

from omniprobe.models.probes import DPT_FPN
from tests.fakes import TinyDenseBackbone


def _script_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "task_name": "detection3d_omni3d",
            "image_mean": "imagenet",
            "output_dir": "/tmp/omniprobe-tests/det3d",
            "system": {"random_seed": 8, "num_gpus": 1, "port": 12356},
            "dataset_root": "/tmp/omni3d",
            "datasets": {
                "train": ["ARKitScenes_train", "ARKitScenes_val"],
                "test": ["ARKitScenes_test"],
                "category_names": ["bed", "table", "chair"],
                "num_classes": 3,
            },
            "solver": {
                "type": "adamw",
                "base_lr": 0.001,
                "ims_per_batch": 4,
                "max_iter": 100,
                "steps": [60, 80],
                "warmup_iters": 10,
                "checkpoint_period": 50,
                "amp": False,
            },
            "test": {"eval_period": 0},
            "freeze_backbone": True,
            "pixel_norm": "cubercnn",
            "stabilize": 0.02,
            "max_attempts": 10,
            "eval_only": False,
            "resume": False,
            "weights": "",
            "visualize_predictions": False,
            "d2_overrides": [],
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_dpt_fpn_pyramid_shapes():
    probe = DPT_FPN(input_dims=[8, 8, 8, 8], output_dim=16, hidden_dim=32)
    feats = [torch.randn(2, 8, 32, 32) for _ in range(4)]
    out = probe(feats)

    assert set(out.keys()) == {"p2", "p3", "p4", "p5"}
    assert out["p2"].shape == (2, 16, 128, 128)
    assert out["p3"].shape == (2, 16, 64, 64)
    assert out["p4"].shape == (2, 16, 32, 32)
    assert out["p5"].shape == (2, 16, 16, 16)
    assert probe.output_dim == 16


def test_dpt_fpn_mixed_input_dims():
    probe = DPT_FPN(input_dims=[4, 8, 12, 16], output_dim=8, hidden_dim=16)
    feats = [torch.randn(1, dim, 16, 16) for dim in (4, 8, 12, 16)]
    out = probe(feats)
    assert out["p2"].shape == (1, 8, 64, 64)
    assert out["p5"].shape == (1, 8, 8, 8)


class _FourLayerDenseBackbone(TinyDenseBackbone):
    """TinyDenseBackbone variant emitting the 4 dense maps the adapter needs."""

    def __init__(self):
        super().__init__(output="dense", return_multilayer=True)
        self.feat_dim = [4, 4, 4, 4]

    def forward(self, images):
        feat = torch.ones(images.shape[0], 4, 2, 2, device=images.device)
        return [feat] * 4


def _make_adapter(freeze=True):
    from omniprobe.models.detectron2_backbone import OmniProbeD2Backbone

    model = _FourLayerDenseBackbone()
    probe = DPT_FPN(input_dims=list(model.feat_dim), output_dim=8, hidden_dim=16)
    return OmniProbeD2Backbone(model, probe, freeze=freeze)


def test_adapter_output_shape_and_forward():
    pytest.importorskip("detectron2")
    adapter = _make_adapter()

    shapes = adapter.output_shape()
    assert set(shapes.keys()) == {"p2", "p3", "p4", "p5"}
    assert [shapes[k].stride for k in ("p2", "p3", "p4", "p5")] == [4, 8, 16, 32]
    assert all(spec.channels == 8 for spec in shapes.values())

    out = adapter(torch.randn(2, 3, 4, 4))
    # TinyDenseBackbone emits 2x2 dense maps regardless of input size
    assert out["p2"].shape == (2, 8, 8, 8)
    assert out["p5"].shape == (2, 8, 1, 1)


def test_adapter_rejects_single_tensor_backbone():
    pytest.importorskip("detectron2")
    from omniprobe.models.detectron2_backbone import OmniProbeD2Backbone

    model = TinyDenseBackbone(output="dense")  # returns one tensor
    probe = DPT_FPN(input_dims=[4, 4, 4, 4], output_dim=8, hidden_dim=16)
    adapter = OmniProbeD2Backbone(model, probe)
    with pytest.raises(TypeError):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_freeze_semantics():
    pytest.importorskip("detectron2")
    adapter = _make_adapter(freeze=True)
    adapter.train()
    assert not adapter.model.training
    assert adapter.probe.training
    assert all(not p.requires_grad for p in adapter.model.parameters())
    assert all(p.requires_grad for p in adapter.probe.parameters())

    unfrozen = _make_adapter(freeze=False)
    unfrozen.train()
    assert unfrozen.model.training


def test_task_script_imports_without_detectron2():
    import omniprobe.scripts.train_detection3d_omni3d as task_module

    assert callable(task_module.run_task)


def test_resolve_category_names_presets():
    from omniprobe.scripts.train_detection3d_omni3d import resolve_category_names

    assert len(resolve_category_names("omni3d")) == 50
    assert len(resolve_category_names("omni3d_in")) == 38
    assert len(resolve_category_names("omni3d_out")) == 11
    assert len(resolve_category_names("KITTI_test")) == 5
    assert resolve_category_names(["bed", "chair"]) == ["bed", "chair"]
    with pytest.raises(ValueError):
        resolve_category_names("not_a_dataset")


def test_build_d2_cfg_category_preset():
    pytest.importorskip("detectron2")
    from omniprobe.scripts.train_detection3d_omni3d import build_d2_cfg

    cfg = _script_cfg()
    cfg.datasets.category_names = "omni3d_out"
    cfg.datasets.num_classes = None
    d2 = build_d2_cfg(cfg)
    assert d2.MODEL.ROI_HEADS.NUM_CLASSES == 11
    assert "traffic cone" in d2.DATASETS.CATEGORY_NAMES


def test_build_d2_cfg_roundtrip():
    pytest.importorskip("detectron2")
    from omniprobe.scripts.train_detection3d_omni3d import build_d2_cfg

    d2 = build_d2_cfg(_script_cfg())
    assert d2.DATASETS.TRAIN == ("ARKitScenes_train", "ARKitScenes_val")
    assert d2.SOLVER.STEPS == (60, 80)
    assert d2.MODEL.ROI_HEADS.NUM_CLASSES == 3
    assert d2.MODEL.PROPOSAL_GENERATOR.NAME == "RPNWithIgnore"
    assert d2.MODEL.ROI_HEADS.NAME == "ROIHeads3D"
    assert d2.MODEL.PIXEL_MEAN == [103.530, 116.280, 123.675]
    assert d2.INPUT.FORMAT == "RGB"
    assert d2.VIS_PERIOD == 0


def test_build_d2_cfg_backbone_pixel_norm():
    pytest.importorskip("detectron2")
    from omniprobe.scripts.train_detection3d_omni3d import build_d2_cfg

    d2 = build_d2_cfg(_script_cfg(pixel_norm="backbone"))
    assert d2.MODEL.PIXEL_MEAN == pytest.approx([0.485 * 255, 0.456 * 255, 0.406 * 255])
    assert d2.MODEL.PIXEL_STD == pytest.approx([0.229 * 255, 0.224 * 255, 0.225 * 255])


def test_build_d2_cfg_visualize_flag():
    pytest.importorskip("detectron2")
    from omniprobe.scripts.train_detection3d_omni3d import build_d2_cfg

    assert build_d2_cfg(_script_cfg()).TEST.VISUALIZE_PREDICTIONS is False
    d2 = build_d2_cfg(_script_cfg(visualize_predictions=True))
    assert d2.TEST.VISUALIZE_PREDICTIONS is True


def test_build_d2_cfg_overrides():
    pytest.importorskip("detectron2")
    from omniprobe.scripts.train_detection3d_omni3d import build_d2_cfg

    d2 = build_d2_cfg(
        _script_cfg(d2_overrides=["SOLVER.WEIGHT_DECAY", "0.05"])
    )
    assert d2.SOLVER.WEIGHT_DECAY == 0.05
