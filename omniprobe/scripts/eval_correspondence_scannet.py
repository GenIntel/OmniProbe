from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from omniprobe.datasets.scannet_pairs import ScanNetPairsDataset
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_depth,
    project_3dto2d,
)
from omniprobe.utils.transformations import so3_rotation_angle, transform_points_Rt


def run_task(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    device = torch.device(cfg.device) if "device" in cfg else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to(device)
    if "data_root" in cfg:
        dataset = ScanNetPairsDataset(root=cfg.data_root, image_mean=cfg.image_mean)
    else:
        dataset = ScanNetPairsDataset(image_mean=cfg.image_mean)
    loader = DataLoader(
        dataset, 8, num_workers=4, drop_last=False, pin_memory=True, shuffle=False
    )

    # extract features
    err_2d = []
    R_gt = []
    for i in tqdm(range(len(dataset))):
        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        deps = torch.stack((instance["depth_0"], instance["depth_1"]), dim=0)
        K_mat = instance["K"].clone()
        Rt_gt = instance["Rt_1"].float()[:3, :4]
        R_gt.append(Rt_gt[:3, :3])

        feats = model(rgbs.to(device))
        if cfg.multilayer:
            feats = torch.cat(feats, dim=1)

        # scale depth and intrinsics
        feats = feats.detach().cpu()
        deps = nn_F.interpolate(deps, scale_factor=cfg.scale_factor)
        K_mat[:2, :] *= cfg.scale_factor

        # compute corr
        corr_xyz0, corr_xyz1, corr_dist = estimate_correspondence_depth(
            feats[0], feats[1], deps[0], deps[1], K_mat.clone(), cfg.num_corr
        )

        # compute error
        corr_xyz0in1 = transform_points_Rt(corr_xyz0, Rt_gt)
        uv_0in1 = project_3dto2d(corr_xyz0in1, K_mat.clone())
        uv_1in1 = project_3dto2d(corr_xyz1, K_mat.clone())
        corr_err2d = (uv_0in1 - uv_1in1).norm(p=2, dim=1)
        err_2d.append(corr_err2d.detach().cpu())

    err_2d = torch.stack(err_2d, dim=0).float()
    R_gt = torch.stack(R_gt, dim=0).float()

    results = []
    # compute 2D errors
    px_thresh = [5, 10, 20]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        print(f"Recall at {_th:>2d} pixels:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")

    # compute rel_ang
    rel_ang = so3_rotation_angle(R_gt)
    rel_ang = rel_ang * 180.0 / np.pi

    # compute thresholded recall
    rec_10px = (err_2d < 10).float().mean(dim=1)
    bin_rec = compute_binned_performance(rec_10px, rel_ang, [0, 15, 30, 60, 180])
    for bin_acc in bin_rec:
        results.append(f"{bin_acc * 100:5.02f}")

    # # result summary
    output_dir = str(HydraConfig.get().run.dir)
    entry = build_result_entry(
        "scannet",
        "default",
        model,
        output_dir,
        cfg,
        {
            "Recall@5px": float(100 * (err_2d < 5).float().mean()),
            "Recall@10px": float(100 * (err_2d < 10).float().mean()),
            "Recall@20px": float(100 * (err_2d < 20).float().mean()),
            "Recall@5deg": float(100 * (rel_ang < 5).float().mean()),
            "Recall@15deg": float(100 * (rel_ang < 15).float().mean()),
            "Recall@45deg": float(100 * (rel_ang < 45).float().mean()),
        },
        dataset=loader.dataset.name,
        num_corr=int(cfg.num_corr),
        scale_factor=float(cfg.scale_factor),
    )
    append_jsonl(
        resolve_results_path(cfg, "correspondence_scannet.jsonl"),
        entry,
    )
