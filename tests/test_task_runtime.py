from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import OmegaConf

from omniprobe.runtime import (
    RuntimeContext,
    configure_run_logging,
)
from omniprobe.tasks import load_task_module
from omniprobe.tasks.script_task import build_script_cfg, run_script_task


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
    "train_detection3d_omni3d",
    "train_pose_imagenet3d",
    "train_segmentation_ade20k",
    "train_snorm",
]


def _context():
    return RuntimeContext(OmegaConf.create({}), torch.device("cpu"), Path("/tmp/omniprobe-tests"))


def _dense_cfg(task_name):
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
                "data_root": "/tmp",
                "train_split": "train",
                "val_split": "val",
                "image_size": 4,
                "batch_size": 2,
                "num_workers": 0,
                "knn_k": [1],
                "temperature": 0.07,
                "output_preference": ["cls", "gap"],
                "result_log": "/tmp/omniprobe-tests.jsonl",
                "soft_eval": False,
                "matching_strategy": "nn",
            },
        }
    )


def _with_runner(cfg, module_name, **runner):
    cfg.task.runner = {"module": module_name, "required_output": "dense", **runner}
    return cfg


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
                "data_root": "/tmp",
                "train_split": "train",
                "val_split": "val",
                "image_size": 4,
                "batch_size": 2,
                "num_workers": 0,
                "knn_k": [1],
                "temperature": 0.07,
                "output_preference": ["cls", "gap"],
                "result_log": "/tmp/omniprobe-tests.jsonl",
            },
        }
    )


def _backbone_cfg(name: str):
    root = Path(__file__).resolve().parents[1]
    return OmegaConf.load(root / "configs" / "backbone" / f"{name}.yaml")


def test_task_configs_compose():
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for path in sorted((config_dir / "task").glob("*.yaml")):
            cfg = hydra.compose(config_name="run", overrides=[f"task={path.stem}"])
            assert cfg.task.name
            assert "image_mean" not in cfg.task
            if cfg.task.name not in {"classification_imagenet_knn", "classification_imagenet_linear"}:
                assert cfg.task.runner.module


@pytest.mark.parametrize(
    ("task_name", "dataset_target"),
    [
        ("correspondence_navi", "omniprobe.datasets.navi.NAVI"),
        ("tracking_tapvid", "omniprobe.datasets.tapvid.TAPVidDataset"),
    ],
)
def test_script_payload_includes_composed_dataset_config(task_name, dataset_target):
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="run",
            overrides=[f"task={task_name}", "backbone=dinov2_b14"],
        )

    script_cfg = build_script_cfg(cfg)
    assert script_cfg.dataset._target_ == dataset_target


def test_default_run_output_root_stays_outputs():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs/run.yaml")
    run_cfg = OmegaConf.to_container(cfg.hydra.run, resolve=False)
    assert str(run_cfg["dir"]).startswith("outputs/")
    assert cfg.results_dir == "results"


def test_task_defaults_override_script_defaults():
    cfg = _dense_cfg("correspondence_spair")
    cfg.task.eval_before_training = False
    cfg.task.num_instances = None
    cfg.task.eval_num_instances = None
    script_cfg = build_script_cfg(cfg)
    assert script_cfg.task_name == "correspondence_spair"
    assert script_cfg.output_dir
    assert script_cfg.eval_before_training is False
    assert script_cfg.num_instances is None
    assert script_cfg.eval_num_instances is None


def test_soco_linear_probe_uses_flat_task_mask_bbox_defaults():
    cfg = _dense_cfg("correspondence_soco")
    cfg.task.mask_feats = False
    cfg.task.use_bbox = False
    script_cfg = build_script_cfg(cfg)
    assert script_cfg.use_bbox is False
    assert script_cfg.mask_feats is False


def test_task_cli_style_overrides_still_win():
    cfg = _dense_cfg("correspondence_soco")
    cfg.task.mask_feats = True
    script_cfg = build_script_cfg(cfg)
    assert script_cfg.mask_feats is True


def test_mode_task_defaults_and_cli_overrides_compose():
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="run",
            overrides=[
                "task=correspondence_spair_linear_probe",
                "task.train.epochs=7",
                "backbone=dinov2_b14",
                "device=cpu",
            ],
        )
        script_cfg = build_script_cfg(cfg)
    assert script_cfg.num_instances is None
    assert script_cfg.eval_num_instances is None
    assert script_cfg.train.epochs == 7


@pytest.mark.parametrize(
    ("backbone_name", "expected_image_mean"),
    [
        ("clip_b16", "clip"),
        ("clip_convnext", "clip"),
        ("perception_b16_512", "perception"),
        ("c_radio_3_b", "raw"),
        ("dinov2_b14", "imagenet"),
    ],
)
def test_script_payload_image_mean_follows_backbone_config(backbone_name, expected_image_mean):
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="run",
            overrides=[
                "task=correspondence_soco",
                f"backbone={backbone_name}",
                "device=cpu",
            ],
        )
        script_cfg = build_script_cfg(cfg)
    assert cfg.backbone.image_mean == expected_image_mean
    assert "image_mean" not in cfg.task
    assert script_cfg.image_mean == expected_image_mean
    assert "image_mean" not in script_cfg.backbone


def test_segmentation_eval_requires_explicit_checkpoint_by_default():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs/task/segmentation_ade20k_eval.yaml")
    assert cfg.checkpoint_path is None
    assert cfg.visualization_dir is None


def test_run_logging_writes_loguru_records(tmp_path):
    with configure_run_logging(tmp_path):
        logger.info("loguru message")

    run_log = (tmp_path / "run.log").read_text()
    assert "loguru message" in run_log
    assert not (tmp_path / "console.log").exists()


def test_segmentation_train_defaults_use_short_schedule():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs/task/segmentation_ade20k.yaml")
    assert cfg.optimizer.max_epochs == 30
    assert cfg.optimizer.drop_at == 20


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
                "data_root": "/tmp",
                "train_split": "train",
                "val_split": "val",
                "image_size": 4,
                "batch_size": 2,
                "num_workers": 0,
                "knn_k": [1],
                "temperature": 0.07,
                "output_preference": ["cls", "gap"],
                "result_log": "/tmp/omniprobe-tests.jsonl",
            },
        }
    )
    module = load_task_module("classification_imagenet_knn")
    with pytest.raises(ValueError, match="does not support any output from task.output_preference"):
        module.run(cfg, _context())


def test_spair_task_smoke(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        module_name,
        extra_overrides=None,
    ):
        calls.append((cfg.task.name, module_name, extra_overrides))
        return {"ok": True}

    import omniprobe.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "run_script_task", fake_run_script_task)
    module = load_task_module("correspondence_spair")
    module.run(
        _with_runner(
            _dense_cfg("correspondence_spair"),
            "omniprobe.scripts.eval_correspondence_spair",
        ),
        _context(),
    )
    spair_soft_cfg = _with_runner(
        _dense_cfg("correspondence_spair"),
        "omniprobe.scripts.eval_correspondence_spair",
    )
    spair_soft_cfg.task.soft_eval = True
    module.run(spair_soft_cfg, _context())
    load_task_module("correspondence_spair_linear_probe").run(
        _with_runner(
            _dense_cfg("correspondence_spair_linear_probe"),
            "omniprobe.scripts.eval_correspondence_linear_probe_spair",
        ),
        _context(),
    )
    assert calls[0] == (
        "correspondence_spair",
        "omniprobe.scripts.eval_correspondence_spair",
        None,
    )
    assert calls[1] == (
        "correspondence_spair",
        "omniprobe.scripts.eval_correspondence_spair",
        None,
    )
    assert calls[2][:2] == (
        "correspondence_spair_linear_probe",
        "omniprobe.scripts.eval_correspondence_linear_probe_spair",
    )


def test_correspondence_soco_task_smoke(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        module_name,
        extra_overrides=None,
    ):
        calls.append((cfg.task.name, module_name, extra_overrides))
        return {"ok": True}

    import omniprobe.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "run_script_task", fake_run_script_task)
    module = load_task_module("correspondence_soco")
    module.run(
        _with_runner(
            _dense_cfg("correspondence_soco"),
            "omniprobe.scripts.eval_correspondence_soco",
        ),
        _context(),
    )
    load_task_module("correspondence_soco_linear_probe").run(
        _with_runner(
            _dense_cfg("correspondence_soco_linear_probe"),
            "omniprobe.scripts.eval_correspondence_linear_probe_soco",
        ),
        _context(),
    )
    assert calls[0] == (
        "correspondence_soco",
        "omniprobe.scripts.eval_correspondence_soco",
        None,
    )
    assert calls[1][:2] == (
        "correspondence_soco_linear_probe",
        "omniprobe.scripts.eval_correspondence_linear_probe_soco",
    )


def test_imagenet3d_pose_mode_smoke(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        module_name,
        extra_overrides=None,
    ):
        calls.append((cfg.task.name, module_name, extra_overrides))
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
                "runner": {
                    "module": "omniprobe.scripts.train_pose_imagenet3d",
                    "required_output": "global",
                },
            },
        }
    )
    module = load_task_module("pose_imagenet3d")
    module.run(cfg, _context())
    cfg.task.name = "pose_imagenet3d_ep"
    cfg.task.runner.required_output = "dense"
    load_task_module("pose_imagenet3d_ep").run(cfg, _context())
    assert calls[0] == (
        "pose_imagenet3d",
        "omniprobe.scripts.train_pose_imagenet3d",
        None,
    )
    assert calls[1] == (
        "pose_imagenet3d_ep",
        "omniprobe.scripts.train_pose_imagenet3d",
        None,
    )


def test_depth_dispatch_forces_multilayer(monkeypatch):
    calls = []

    def fake_run_script_task(
        cfg,
        module_name,
        extra_overrides=None,
    ):
        calls.append((cfg.task.name, module_name, extra_overrides))
        return {"ok": True}

    import omniprobe.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "run_script_task", fake_run_script_task)
    module = load_task_module("depth")
    cfg = _with_runner(
        _dense_cfg("depth"),
        "omniprobe.scripts.train_depth",
        require_multilayer=True,
    )
    module.run(cfg, _context())
    assert calls == [
        (
            "depth",
            "omniprobe.scripts.train_depth",
            {"backbone": {"return_multilayer": True}},
        )
    ]


def test_build_script_cfg_preserves_backbone_overrides():
    cfg = _dense_cfg("depth")
    script_cfg = build_script_cfg(
        cfg,
        extra_overrides={"backbone": {"return_multilayer": True}},
    )
    assert script_cfg.backbone.return_multilayer is True


def test_build_script_cfg_replaces_legacy_backbone_keys():
    cfg = _dense_cfg("spair")
    cfg.backbone = _backbone_cfg("c_radio_3_b")
    script_cfg = build_script_cfg(cfg)
    assert script_cfg.backbone._target_ == "omniprobe.models.c_radio.CRADIOv3Backbone"
    assert script_cfg.backbone.version == "c-radio_v3-b"
    assert "dino_name" not in script_cfg.backbone
    assert "model_name" not in script_cfg.backbone


@pytest.mark.parametrize("backbone_name", ["clip_b16", "dino_b16", "dinov2_b14", "c_radio_3_b"])
def test_build_script_cfg_accepts_multiple_backbones(backbone_name):
    cfg = _dense_cfg("spair")
    cfg.backbone = _backbone_cfg(backbone_name)
    script_cfg = build_script_cfg(cfg)
    assert script_cfg.backbone._target_ == cfg.backbone._target_


def test_build_script_cfg_preserves_task_root_overrides():
    cfg = _dense_cfg("spair")
    cfg.task.data_root = "/tmp/custom-spair"
    script_cfg = build_script_cfg(cfg)
    assert script_cfg.data_root == "/tmp/custom-spair"


def test_build_script_cfg_resolves_auto_device(monkeypatch):
    monkeypatch.setattr("omniprobe.tasks.script_task.resolve_device", lambda device_name: torch.device("cpu"))
    cfg = _dense_cfg("navi")
    cfg.device = "auto"
    script_cfg = build_script_cfg(cfg)
    assert script_cfg.device == "cpu"


def test_run_script_task_dispatches_via_run_task(monkeypatch):
    seen = []

    def fake_import_module(module_name):
        assert module_name == "fake_script"
        return SimpleNamespace(
            run_task=lambda cfg: seen.append(cfg) or {"ok": True, "device": str(cfg.device)}
        )

    monkeypatch.setattr("omniprobe.tasks.script_task.import_module", fake_import_module)
    cfg = _dense_cfg("spair")
    result = run_script_task(cfg, "fake_script")
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


def test_imagenet_knn_prefers_cls_over_gap():
    import omniprobe.tasks.imagenet_knn as task_module
    from omniprobe.models.contracts import get_backbone_contract

    contract = get_backbone_contract(_global_cfg().backbone)
    task_cfg = OmegaConf.create({"knn_k": [1], "output_preference": ["cls", "gap"]})
    assert task_module._resolve_knn_output(contract, task_cfg) == "cls"


def test_imagenet_knn_falls_back_when_cls_is_unavailable():
    import omniprobe.tasks.imagenet_knn as task_module
    from omniprobe.models.contracts import get_backbone_contract

    contract = get_backbone_contract(_dense_cfg("classification_imagenet_knn").backbone)
    task_cfg = OmegaConf.create({"knn_k": [1], "output_preference": ["cls", "gap"]})
    assert task_module._resolve_knn_output(contract, task_cfg) == "gap"


def test_imagenet_knn_respects_configured_output_preference():
    import omniprobe.tasks.imagenet_knn as task_module
    from omniprobe.models.contracts import get_backbone_contract

    contract = get_backbone_contract(_global_cfg().backbone)
    task_cfg = OmegaConf.create({"knn_k": [1], "output_preference": ["gap", "cls"]})
    assert task_module._resolve_knn_output(contract, task_cfg) == "gap"


def test_imagenet_loaders_use_backbone_image_mean(monkeypatch):
    import omniprobe.tasks.imagenet_common as imagenet_common

    seen = []

    def fake_build_dataloader(data_cfg, train):
        seen.append((data_cfg.mean, data_cfg.std, train))
        return SimpleNamespace(dataset=SimpleNamespace(classes=["a"]))

    monkeypatch.setattr(
        imagenet_common,
        "build_imagenet_dataloader",
        fake_build_dataloader,
    )
    task_cfg = OmegaConf.create(
        {
            "data_root": "/tmp",
            "train_split": "train",
            "val_split": "val",
            "image_size": 4,
            "batch_size": 2,
            "num_workers": 0,
        }
    )
    backbone_cfg = OmegaConf.create({"image_mean": "clip"})
    imagenet_common.build_imagenet_loaders(task_cfg, backbone_cfg)
    assert seen == [
        (
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
            True,
        ),
        (
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
            False,
        ),
    ]


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
