from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from omniprobe.datasets.builder import build_loader
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_xyz,
    project_3dto2d,
)
from omniprobe.utils.progress import progress
from omniprobe.utils.transformations import so3_rotation_angle, transform_points_Rt


def run_task(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    device = torch.device(cfg.device) if "device" in cfg else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to(device)
    loader = build_loader(cfg.dataset, "test", 4, 1, pair_dataset=True)
    _ = loader.dataset.__getitem__(0)

    # extract features
    feats_0 = []
    feats_1 = []
    xyz_grid_0 = []
    xyz_grid_1 = []
    Rt_gt = []
    intrinsics = []

    for batch in progress(loader, desc="NAVI feature extraction"):
        feat_0 = model(batch["image_0"].to(device))
        feat_1 = model(batch["image_1"].to(device))
        if cfg.multilayer:
            feat_0 = torch.cat(feat_0, dim=1)
            feat_1 = torch.cat(feat_1, dim=1)
        feats_0.append(feat_0.detach().cpu())
        feats_1.append(feat_1.detach().cpu())
        Rt_gt.append(batch["Rt_01"])
        intrinsics.append(batch["intrinsics_1"])

        # scale down to avoid a huge matching problem
        xyz_grid_0_i = nn_F.interpolate(
            batch["xyz_grid_0"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_1_i = nn_F.interpolate(
            batch["xyz_grid_1"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_0.append(xyz_grid_0_i)
        xyz_grid_1.append(xyz_grid_1_i)

    feats_0 = torch.cat(feats_0, dim=0)
    feats_1 = torch.cat(feats_1, dim=0)
    xyz_grid_0 = torch.cat(xyz_grid_0, dim=0)
    xyz_grid_1 = torch.cat(xyz_grid_1, dim=0)
    Rt_gt = torch.cat(Rt_gt, dim=0).float()[:, :3, :4]
    intrinsics = torch.cat(intrinsics, dim=0).float()

    num_instances = len(loader.dataset)
    err_3d = []
    err_2d = []
    for i in progress(range(num_instances), desc="NAVI correspondence evaluation"):
        c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1 = estimate_correspondence_xyz(
            feats_0[i], feats_1[i], xyz_grid_0[i], xyz_grid_1[i], cfg.num_corr
        )

        c_uv0 = c_uv0 / cfg.scale_factor
        c_uv1 = c_uv1 / cfg.scale_factor

        c_xyz0in1 = transform_points_Rt(c_xyz0, Rt_gt[i].float())
        c_err3d = (c_xyz0in1 - c_xyz1).norm(p=2, dim=1)

        c_xyz1in1_uv = project_3dto2d(c_xyz1, intrinsics[i])
        c_xyz0in1_uv = project_3dto2d(c_xyz0in1, intrinsics[i])
        c_err2d = (c_xyz0in1_uv - c_xyz1in1_uv).norm(p=2, dim=1)

        err_3d.append(c_err3d.detach().cpu())
        err_2d.append(c_err2d.detach().cpu())

    err_3d = torch.stack(err_3d, dim=0).float()
    err_2d = torch.stack(err_2d, dim=0).float()
    results = []
    perf_bins_3d = {}

    metric_thresh = [0.01, 0.02, 0.05]
    for _th in metric_thresh:
        recall_i = 100 * (err_3d < _th).float().mean()
        logger.info(f"Recall at {_th:>.2f} m:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")
        perf_bins_3d[_th] = float(recall_i / 100.0)

    perf_bins_2d = {}
    px_thresh = [5, 25, 50]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        logger.info(f"Recall at {_th:>3d}px:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")
        perf_bins_2d[_th] = float(recall_i / 100.0)

    # compute rel_ang
    rel_ang = so3_rotation_angle(Rt_gt[:, :3, :3])
    rel_ang = rel_ang * 180.0 / np.pi

    # compute thresholded recall -- 0.2decimeter = 2cm
    rec_2cm = (err_3d < 0.02).float().mean(dim=1)
    angle_bins = [0, 30, 60, 90, 120]
    bin_rec = compute_binned_performance(rec_2cm, rel_ang, angle_bins)
    for bin_acc in bin_rec:
        results.append(f"{bin_acc * 100:5.02f}")
    perf_bins_ang = {}
    for idx, bin_acc in enumerate(bin_rec):
        key = f"{angle_bins[idx]}-{angle_bins[idx + 1]}deg@2cm"
        perf_bins_ang[key] = float(bin_acc)

    # # result summary
    dset = loader.dataset.name
    output_dir = str(HydraConfig.get().run.dir)
    entry = build_result_entry(
        "navi",
        model,
        output_dir,
        cfg,
        {
            "Recall@0.01m": perf_bins_3d[0.01] * 100,
            "Recall@0.02m": perf_bins_3d[0.02] * 100,
            "Recall@0.05m": perf_bins_3d[0.05] * 100,
            "Recall@5px": perf_bins_2d[5] * 100,
            "Recall@25px": perf_bins_2d[25] * 100,
            "Recall@50px": perf_bins_2d[50] * 100,
            "Recall@0-30deg@2cm": perf_bins_ang["0-30deg@2cm"] * 100,
            "Recall@30-60deg@2cm": perf_bins_ang["30-60deg@2cm"] * 100,
            "Recall@60-90deg@2cm": perf_bins_ang["60-90deg@2cm"] * 100,
            "Recall@90-120deg@2cm": perf_bins_ang["90-120deg@2cm"] * 100,
        },
        dataset=dset,
        num_corr=int(cfg.num_corr),
        scale_factor=float(cfg.scale_factor),
    )
    append_jsonl(resolve_results_path(cfg, "correspondence_navi.jsonl"), entry)
