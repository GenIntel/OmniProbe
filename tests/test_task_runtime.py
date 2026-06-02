from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from omniprobe.runtime import RuntimeContext
from omniprobe.tasks import load_task_module
from omniprobe.tasks.script_task import build_script_cfg, load_script_config, run_script_task


_SCRIPT_BACKENDS = [
    "eval_correspondence_ap10k",
    "eval_correspondence_geometric_soco",
    "eval_correspondence_linear_probe_soco",
    "eval_correspondence_linear_probe_spair",
    "eval_correspondence_navi",
    "eval_correspondence_scannet",
    "eval_correspondence_soco",
    "eval_correspondence_spair",
    "eval_segmentation_ade20k",
    "eval_tracking_tapvid",
    "train_depth",
    "train_pose_imagenet3d",
    "train_segmentation_ade20k",
    "train_snorm",
]

_RESULT_METADATA_FILES = [
    "omniprobe/tasks/imagenet_knn.py",
    "omniprobe/tasks/imagenet_linear.py",
    "omniprobe/scripts/eval_correspondence_ap10k.py",
    "omniprobe/scripts/eval_correspondence_geometric_soco.py",
    "omniprobe/scripts/eval_correspondence_linear_probe_soco.py",
    "omniprobe/scripts/eval_correspondence_linear_probe_spair.py",
    "omniprobe/scripts/eval_correspondence_navi.py",
    "omniprobe/scripts/eval_correspondence_scannet.py",
    "omniprobe/scripts/eval_correspondence_soco.py",
    "omniprobe/scripts/eval_correspondence_spair.py",
    "omniprobe/scripts/eval_segmentation_ade20k.py",
    "omniprobe/scripts/eval_tracking_tapvid.py",
    "omniprobe/scripts/train_depth.py",
    "omniprobe/scripts/train_pose_imagenet3d.py",
    "omniprobe/scripts/train_segmentation_ade20k.py",
    "omniprobe/scripts/train_snorm.py",
]


def _context():
    return RuntimeContext(OmegaConf.create({}), torch.device("cpu"), Path("/tmp/omniprobe-tests"))


def _dense_cfg(task_name, mode_name):
    return OmegaConf.create(
        {
            "backbone": {
                "_target_": "tests.fakes.TinyDenseBackbone",
                "supported_outputs": ["dense", "gap"],
                "default_global_output": "gap",
                "supports_multilayer": True,
                "supports_layer_selection": True,
                "image_mean": "imagenet",
            },
            "device": "cpu",
            "task": {
                "name": task_name,
                "mode": mode_name,
                "image_mean": "imagenet",
                "data_root": "/tmp",
                "train_split": "train",
                "val_split": "val",
                "image_size": 4,
                "batch_size": 2,
                "num_workers": 0,
                "knn_k": [1],
                "temperature": 0.07,
                "result_log": "/tmp/omniprobe-tests.jsonl",
                "soft_eval": False,
                "matching_strategy": "nn",
            },
        }
    )


def _global_cfg():
    return OmegaConf.create(
        {
            "backbone": {
                "_target_": "tests.fakes.TinyGlobalBackbone",
                "supported_outputs": ["cls", "gap"],
                "default_global_output": "cls",
                "supports_multilayer": False,
                "supports_layer_selection": True,
                "image_mean": "imagenet",
            },
            "task": {
                "name": "classification_imagenet_knn",
                "mode": "default",
                "data_root": "/tmp",
                "train_split": "train",
                "val_split": "val",
                "image_size": 4,
                "batch_size": 2,
                "num_workers": 0,
                "image_mean": "imagenet",
                "knn_k": [1],
                "temperature": 0.07,
                "result_log": "/tmp/omniprobe-tests.jsonl",
            },
        }
    )


def _backbone_cfg(name: str):
    root = Path(__file__).resolve().parents[1]
    return OmegaConf.load(root / "configs" / "backbone" / f"{name}.yaml")


def test_invalid_task_mode_fails_clearly():
    cfg = _dense_cfg("correspondence_spair", "hungarian")
    module = load_task_module("correspondence_spair")
    with pytest.raises(ValueError, match="Unsupported mode"):
        module.run(cfg, _context())


def test_script_config_defaults_are_composed():
    cfg = load_script_config("eval_tracking_tapvid")
    assert "backbone" in cfg
    assert "dataset" in cfg



def test_spair_linear_probe_keeps_legacy_num_instances_default():
    cfg = _dense_cfg("correspondence_spair", "linear_probe")
    script_cfg = build_script_cfg(cfg, "correspondence_spair", "eval_correspondence_linear_probe_spair")
    assert script_cfg.num_instances == 1000


def test_soco_linear_probe_keeps_mask_bbox_defaults():
    cfg = _dense_cfg("correspondence_soco", "linear_probe")
    script_cfg = build_script_cfg(
        cfg,
        "correspondence_soco",
        "eval_correspondence_linear_probe_soco",
    )
    assert script_cfg.use_bbox is False
    assert script_cfg.mask_feats is True


def test_invalid_task_backbone_combo_fails_clearly():
    cfg = OmegaConf.create(
        {
            "backbone": {
                "_target_": "tests.fakes.TinyDenseBackbone",
                "supported_outputs": ["dense"],
                "supports_multilayer": True,
                "supports_layer_selection": True,
                "image_mean": "imagenet",
            },
            "task": {
                "name": "classification_imagenet_knn",
                "mode": "default",
                "data_root": "/tmp",
                "train_split": "train",
                "val_split": "val",
                "image_size": 4,
                "batch_size": 2,
                "num_workers": 0,
                "image_mean": "imagenet",
                "knn_k": [1],
                "temperature": 0.07,
                "result_log": "/tmp/omniprobe-tests.jsonl",
            },
        }
    )
    module = load_task_module("classification_imagenet_knn")
    with pytest.raises(ValueError, match="does not expose a global output"):
        module.run(cfg, _context())


def test_spair_mode_smoke(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        task_name,
        config_name,
        module_name,
        extra_overrides=None,
    ):
        calls.append((task_name, config_name, module_name, extra_overrides))
        return {"ok": True}

    import omniprobe.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "run_script_task", fake_run_script_task)
    module = load_task_module("correspondence_spair")
    module.run(_dense_cfg("correspondence_spair", "nn"), _context())
    module.run(_dense_cfg("correspondence_spair", "soft_argmax"), _context())
    module.run(_dense_cfg("correspondence_spair", "linear_probe"), _context())
    assert calls[0] == (
        "correspondence_spair",
        "eval_correspondence_spair",
        "omniprobe.scripts.eval_correspondence_spair",
        {"soft_eval": False},
    )
    assert calls[1] == (
        "correspondence_spair",
        "eval_correspondence_spair",
        "omniprobe.scripts.eval_correspondence_spair",
        {"soft_eval": True},
    )
    assert calls[2][:3] == (
        "correspondence_spair",
        "eval_correspondence_linear_probe_spair",
        "omniprobe.scripts.eval_correspondence_linear_probe_spair",
    )


def test_correspondence_soco_mode_smoke(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        task_name,
        config_name,
        module_name,
        extra_overrides=None,
    ):
        calls.append((task_name, config_name, module_name, extra_overrides))
        return {"ok": True}

    import omniprobe.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "run_script_task", fake_run_script_task)
    module = load_task_module("correspondence_soco")
    module.run(_dense_cfg("correspondence_soco", "nn"), _context())
    module.run(_dense_cfg("correspondence_soco", "linear_probe"), _context())
    assert calls[0] == (
        "correspondence_soco",
        "eval_correspondence_soco",
        "omniprobe.scripts.eval_correspondence_soco",
        {"soft_eval": False},
    )
    assert calls[1][:3] == (
        "correspondence_soco",
        "eval_correspondence_linear_probe_soco",
        "omniprobe.scripts.eval_correspondence_linear_probe_soco",
    )


def test_imagenet3d_pose_mode_smoke(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        task_name,
        config_name,
        module_name,
        extra_overrides=None,
    ):
        calls.append((task_name, config_name, module_name, extra_overrides))
        return {"ok": True}

    import omniprobe.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "run_script_task", fake_run_script_task)
    cfg = OmegaConf.create(
        {
            "backbone": {
                "_target_": "tests.fakes.TinyDenseBackbone",
                "supported_outputs": ["dense", "gap"],
                "default_global_output": "gap",
                "supports_multilayer": True,
                "supports_layer_selection": True,
                "image_mean": "imagenet",
            },
            "task": {
                "name": "pose_imagenet3d",
                "mode": "default",
            },
        }
    )
    module = load_task_module("pose_imagenet3d")
    module.run(cfg, _context())
    cfg.task.mode = "ep"
    module.run(cfg, _context())
    assert calls[0] == (
        "pose_imagenet3d",
        "train_pose_imagenet3d",
        "omniprobe.scripts.train_pose_imagenet3d",
        None,
    )
    assert calls[1] == (
        "pose_imagenet3d",
        "train_pose_ep_imagenet3d",
        "omniprobe.scripts.train_pose_imagenet3d",
        None,
    )


def test_depth_dispatch_forces_multilayer(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        task_name,
        config_name,
        module_name,
        extra_overrides=None,
    ):
        calls.append((task_name, config_name, module_name, extra_overrides))
        return {"ok": True}

    import omniprobe.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "run_script_task", fake_run_script_task)
    module = load_task_module("depth")
    cfg = _dense_cfg("depth", "default")
    module.run(cfg, _context())
    assert calls == [
        (
            "depth",
            "train_depth",
            "omniprobe.scripts.train_depth",
            {"backbone": {"return_multilayer": True}},
        )
    ]


def test_build_script_cfg_preserves_backbone_overrides():
    cfg = _dense_cfg("depth", "default")
    script_cfg = build_script_cfg(
        cfg,
        "depth",
        "train_depth",
        extra_overrides={"backbone": {"return_multilayer": True}},
    )
    assert script_cfg.backbone.return_multilayer is True


def test_build_script_cfg_replaces_legacy_backbone_keys():
    cfg = _dense_cfg("spair", "nn")
    cfg.backbone = _backbone_cfg("c_radio_3_b")
    script_cfg = build_script_cfg(cfg, "spair", "eval_correspondence_spair")
    assert script_cfg.backbone._target_ == "omniprobe.models.c_radio.CRADIOv3Backbone"
    assert script_cfg.backbone.version == "c-radio_v3-b"
    assert "dino_name" not in script_cfg.backbone
    assert "model_name" not in script_cfg.backbone


@pytest.mark.parametrize("backbone_name", ["clip_b16", "dino_b16", "dinov2_b14", "c_radio_3_b"])
def test_build_script_cfg_accepts_multiple_backbones(backbone_name):
    cfg = _dense_cfg("spair", "nn")
    cfg.backbone = _backbone_cfg(backbone_name)
    script_cfg = build_script_cfg(cfg, "spair", "eval_correspondence_spair")
    assert script_cfg.backbone._target_ == cfg.backbone._target_


def test_build_script_cfg_preserves_task_root_overrides():
    cfg = _dense_cfg("spair", "nn")
    cfg.task.data_root = "/tmp/custom-spair"
    script_cfg = build_script_cfg(cfg, "spair", "eval_correspondence_spair")
    assert script_cfg.data_root == "/tmp/custom-spair"


def test_build_script_cfg_resolves_auto_device(monkeypatch):
    monkeypatch.setattr("omniprobe.tasks.script_task.resolve_device", lambda device_name: torch.device("cpu"))
    cfg = _dense_cfg("navi", "default")
    cfg.device = "auto"
    script_cfg = build_script_cfg(cfg, "navi", "eval_correspondence_navi")
    assert script_cfg.device == "cpu"


def test_run_script_task_dispatches_via_run_task(monkeypatch):
    seen = []

    def fake_import_module(module_name):
        assert module_name == "fake_script"
        return SimpleNamespace(
            run_task=lambda cfg: seen.append(cfg) or {"ok": True, "device": str(cfg.device)}
        )

    monkeypatch.setattr("omniprobe.tasks.script_task.import_module", fake_import_module)
    cfg = _dense_cfg("spair", "nn")
    result = run_script_task(cfg, "spair", "eval_correspondence_spair", "fake_script")
    assert result["ok"] is True
    assert result["device"] == "cpu"
    assert len(seen) == 1


def test_imagenet_knn_smoke(monkeypatch):
    class TinyDataset:
        classes = ["a", "b", "c", "d", "e"]

    class TinyLoader:
        def __init__(self):
            self.dataset = TinyDataset()
            self._batches = [
                (torch.ones(2, 3, 4, 4), torch.tensor([0, 1])),
                (torch.ones(2, 3, 4, 4), torch.tensor([0, 1])),
            ]

        def __iter__(self):
            return iter(self._batches)

    import omniprobe.tasks.imagenet_knn as task_module

    monkeypatch.setattr(
        task_module,
        "build_imagenet_loaders",
        lambda task_cfg, contract: (TinyLoader(), TinyLoader()),
    )
    result = task_module.run(_global_cfg(), _context())
    assert result["task"] == "imagenet_knn"
    assert result["output"] == "cls"
    assert result["output_dir"] == "/tmp/omniprobe-tests"
    assert isinstance(result["config"], str)


def test_imagenet_knn_extract_features_uses_inference_mode():
    class TinyLoader:
        def __iter__(self):
            yield torch.ones(2, 3, 4, 4), torch.tensor([0, 1])

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(3, 4, kernel_size=1)

        def forward(self, images):
            return self.proj(images)

    import omniprobe.tasks.imagenet_knn as task_module

    features, labels = task_module._extract_features(
        TinyModel(),
        TinyLoader(),
        torch.device("cpu"),
    )
    assert features.requires_grad is False
    assert labels.tolist() == [0, 1]


def test_imagenet_knn_predict_uses_faiss():
    import omniprobe.tasks.imagenet_knn as task_module

    train_feats = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    train_labels = torch.tensor([0, 1, 2])
    val_feats = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    val_labels = torch.tensor([0, 1])

    results = task_module._knn_predict(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        [1, 2],
        0.1,
        3,
    )
    assert results[1]["top1"] == 100.0


def test_script_backends_export_run_task():
    from importlib import import_module as _imp
    for module_name in _SCRIPT_BACKENDS:
        module = _imp(f"omniprobe.scripts.{module_name}")
        assert hasattr(module, "run_task"), module_name



def test_result_overview_writers_include_output_dir_and_config():
    root = Path(__file__).resolve().parents[1]
    for rel_path in _RESULT_METADATA_FILES:
        text = (root / rel_path).read_text()
        assert "append_jsonl" in text, rel_path
        assert "build_result_entry" in text, rel_path
