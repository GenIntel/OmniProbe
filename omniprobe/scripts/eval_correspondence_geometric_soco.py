"""Evaluate geometric SOCO correspondence with concept-restricted candidate patches.

Given a source keypoint, predictions are restricted to target patches that contain
target keypoints with the same concept ID (via ``meta["concept_map"]``).
"""


from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from omniprobe.datasets.soco import SOCODataset
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.scripts.eval_correspondence_soco import _resolve_effective_image_size, set_seed


def _valid_index(idx: int, size: int) -> bool:
    return 0 <= int(idx) < int(size)


def _kp_to_patch_index(kp_xy_norm: torch.Tensor, feat_h: int, feat_w: int) -> int:
    x = float(kp_xy_norm[0].item())
    y = float(kp_xy_norm[1].item())
    px = min(max(int(x * feat_w), 0), feat_w - 1)
    py = min(max(int(y * feat_h), 0), feat_h - 1)
    return py * feat_w + px


@torch.no_grad()
def compute_predictions_geometric(model, instance, mask_feats: bool = False):
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, meta = instance

    device = next(model.parameters()).device
    images = torch.stack((img_i, img_j)).to(device)
    masks = torch.stack((mask_i, mask_j)).to(device)

    feats = model(images)
    if isinstance(feats, list):
        feats = torch.cat(feats, dim=1)
    feats = nn_F.normalize(feats, p=2, dim=1)

    if mask_feats:
        masks = nn_F.interpolate(masks.float(), size=feats.shape[-2:], mode="nearest")
        feats = feats * (masks > 0.5)

    feats_i = feats[0]
    feats_j = feats[1]
    feat_h, feat_w = int(feats_j.shape[-2]), int(feats_j.shape[-1])

    kps_i = kps_i.float().clone()
    kps_j = kps_j.float().clone()
    size = images.shape[-1]
    kps_i[:, :2] = kps_i[:, :2] / size
    kps_j[:, :2] = kps_j[:, :2] / size

    # Source keypoint features and full source->target similarity map.
    kps_i_ndc = (kps_i[:, :2] * 2 - 1)[None, None].to(device)
    kp_i_feats = nn_F.grid_sample(
        feats_i[None, :], kps_i_ndc, mode="bilinear", align_corners=True
    )[0, :, 0].t()
    heatmaps = torch.einsum("kf,fhw->khw", kp_i_feats, feats_j)  # (num_src, H, W)

    pred = torch.full((kps_i.shape[0], 2), -1.0, dtype=torch.float32, device="cpu")
    pred_patch = torch.full((kps_i.shape[0],), -1, dtype=torch.long, device="cpu")
    per_src_candidates: Dict[int, list[int]] = {}
    concept_map = meta.get("concept_map", {})

    for src_idx in range(kps_i.shape[0]):
        if kps_i[src_idx, 2] <= 0:
            continue

        tgt_indices_raw = concept_map.get(int(src_idx), [])
        if not tgt_indices_raw:
            per_src_candidates[int(src_idx)] = []
            continue

        patch_candidates = set()
        valid_tgts = []
        for tgt_idx in tgt_indices_raw:
            if not _valid_index(int(tgt_idx), kps_j.shape[0]):
                continue
            if kps_j[int(tgt_idx), 2] <= 0:
                continue
            valid_tgts.append(int(tgt_idx))
            patch_candidates.add(
                _kp_to_patch_index(kps_j[int(tgt_idx), :2], feat_h=feat_h, feat_w=feat_w)
            )

        per_src_candidates[int(src_idx)] = valid_tgts
        if not patch_candidates:
            continue

        cand_patch = sorted(patch_candidates)
        cand_patch_tensor = torch.tensor(cand_patch, dtype=torch.long, device=device)
        src_scores = heatmaps[src_idx].reshape(-1)
        best_patch_idx = int(cand_patch_tensor[torch.argmax(src_scores[cand_patch_tensor])].item())

        x = best_patch_idx % feat_w
        y = best_patch_idx // feat_w
        pred[src_idx, 0] = float(x) / float(feat_w)
        pred[src_idx, 1] = float(y) / float(feat_h)
        pred_patch[src_idx] = int(best_patch_idx)

    return {
        "pred": pred,
        "pred_patch": pred_patch,
        "gt_src": kps_i,
        "gt_trg": kps_j,
        "thresh_scale": thresh_scale,
        "meta": meta,
        "per_src_candidates": per_src_candidates,
    }


def evaluate_geometric_matches(pred_output, thresh: float):
    pred = pred_output["pred"]
    kps_i = pred_output["gt_src"]
    kps_j = pred_output["gt_trg"]
    scale = float(pred_output["thresh_scale"])
    meta = pred_output["meta"]
    per_src_candidates = pred_output["per_src_candidates"]

    sem_lookup = {int(src_idx): int(tgt_idx) for src_idx, tgt_idx in meta.get("semantic_pairs", [])}

    only_multi_candidate_eval = bool(pred_output.get("only_multi_candidate_eval", False))

    errors = []
    stats = {
        "total_src": 0,
        "skipped_no_corresponding_tgt": 0,
        "skipped_gt_not_in_candidates": 0,
        "skipped_no_candidates": 0,
        "pred_missing": 0,
        "filtered_too_few_candidates": 0,
        "evaluated_src": 0,
    }

    for src_idx in range(kps_i.shape[0]):
        if kps_i[src_idx, 2] <= 0:
            continue
        stats["total_src"] += 1

        gt_tgt = sem_lookup.get(int(src_idx), None)
        if gt_tgt is None or not _valid_index(gt_tgt, kps_j.shape[0]) or kps_j[gt_tgt, 2] <= 0:
            # Evaluate only source keypoints with an actual corresponding target keypoint.
            stats["skipped_no_corresponding_tgt"] += 1
            continue

        candidates = per_src_candidates.get(int(src_idx), [])
        if only_multi_candidate_eval and len(candidates) < 2:
            stats["filtered_too_few_candidates"] += 1
            continue

        if not candidates:
            # Explicitly do not count "no_candidates" as wrong.
            stats["skipped_no_candidates"] += 1
            continue

        if gt_tgt not in candidates:
            stats["skipped_gt_not_in_candidates"] += 1
            continue

        if pred[src_idx, 0] < 0 or pred[src_idx, 1] < 0:
            stats["pred_missing"] += 1
            errors.append(float("inf"))
            continue

        err = (pred[src_idx] - kps_j[gt_tgt, :2]).norm(p=2).item()
        errors.append(err / scale)
        stats["evaluated_src"] += 1

    err_tensor = torch.tensor(errors, dtype=torch.float32) if errors else torch.tensor([])
    pck = (err_tensor < float(thresh)).float().mean().item() * 100.0 if err_tensor.numel() > 0 else float("nan")
    return pck, err_tensor, stats


def evaluate_dataset_geometric(
    model,
    dataset,
    thresh: float,
    mask_feats: bool = False,
    only_multi_candidate_eval: bool = False,
):
    all_errors = []
    total_stats = {
        "total_src": 0,
        "skipped_no_corresponding_tgt": 0,
        "skipped_gt_not_in_candidates": 0,
        "skipped_no_candidates": 0,
        "pred_missing": 0,
        "filtered_too_few_candidates": 0,
        "evaluated_src": 0,
    }

    for idx in range(len(dataset)):
        pred_output = compute_predictions_geometric(model, dataset.__getitem__(idx), mask_feats=mask_feats)
        pred_output["only_multi_candidate_eval"] = only_multi_candidate_eval
        _, err_tensor, stats = evaluate_geometric_matches(pred_output, thresh)
        if err_tensor.numel() > 0:
            all_errors.append(err_tensor)
        for key in total_stats:
            total_stats[key] += int(stats[key])

    if not all_errors:
        return float("nan"), total_stats

    errs = torch.cat(all_errors)
    pck = (errs < float(thresh)).float().mean().item() * 100.0
    return pck, total_stats


def run_task(cfg: DictConfig):
    set_seed(int(cfg.random_seed))
    output_dir = HydraConfig.get().run.dir
    logger.info("Output dir: {}", output_dir)

    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image_size_info = _resolve_effective_image_size(cfg, model)
    effective_image_size = int(image_size_info["image_size"])
    logger.info(
        "Image resolution: {} (fixed_patched_size={}, num_patches={}, resolved_patch_size={}, patch_size_source={}, verified_grid_hw={})",
        effective_image_size,
        bool(cfg.get("fixed_patched_size", False)),
        int(cfg.get("num_patches", 60)),
        image_size_info["patch_size"],
        image_size_info["patch_size_source"],
        image_size_info["verified_grid_hw"],
    )

    data_root = cfg.data_root
    pair_subdir = cfg.pair_subdir
    pair_root = Path(pair_subdir)
    if not pair_root.is_absolute():
        pair_root = Path(data_root) / pair_subdir
    all_classes = sorted([d.name for d in pair_root.iterdir() if d.is_dir()])
    if isinstance(cfg.classes, str) and cfg.classes == "all":
        classes = all_classes
    elif isinstance(cfg.classes, str):
        classes = [cfg.classes]
    elif isinstance(cfg.classes, ListConfig):
        classes = list(cfg.classes)
    else:
        classes = cfg.classes
    if classes is None:
        classes = all_classes

    thresh = float(cfg.thresh)
    class_scores: Dict[str, float] = {}
    aggregate_stats = {
        "total_src": 0,
        "skipped_no_corresponding_tgt": 0,
        "skipped_gt_not_in_candidates": 0,
        "skipped_no_candidates": 0,
        "pred_missing": 0,
        "filtered_too_few_candidates": 0,
        "evaluated_src": 0,
    }

    for class_name in classes:
        dataset = SOCODataset(
            root=data_root,
            image_size=effective_image_size,
            image_mean=cfg.image_mean,
            use_bbox=cfg.use_bbox,
            class_name=class_name,
            max_pairs=cfg.max_pairs,
            pair_subdir=pair_subdir,
        )
        if len(dataset) == 0:
            logger.warning("No pairs for class {}", class_name)
            continue

        pck, stats = evaluate_dataset_geometric(
            model=model,
            dataset=dataset,
            thresh=thresh,
            mask_feats=bool(cfg.mask_feats),
            only_multi_candidate_eval=bool(cfg.only_multi_candidate_eval),
        )
        class_scores[class_name] = pck
        for key in aggregate_stats:
            aggregate_stats[key] += int(stats[key])

        logger.info(
            "PCK@{:.2f} {:20s} | {:6.2f} | src={} | eval={} | skip_no_gt={} | skip_no_cand={} | skip_gt_not_in_cands={} | filtered={} | pred_missing={}",
            thresh,
            class_name,
            pck,
            stats["total_src"],
            stats["evaluated_src"],
            stats["skipped_no_corresponding_tgt"],
            stats["skipped_no_candidates"],
            stats["skipped_gt_not_in_candidates"],
            stats["filtered_too_few_candidates"],
            stats["pred_missing"],
        )

    if not class_scores:
        logger.error("No results computed")
        return

    values = [v for v in class_scores.values() if not np.isnan(v)]
    mean_pck = float(np.mean(values)) if values else float("nan")
    logger.info(
        "Mean PCK@{:.2f} | {:6.2f} | src={} | eval={} | skip_no_gt={} | skip_no_cand={} | skip_gt_not_in_cands={} | filtered={} | pred_missing={}",
        thresh,
        mean_pck,
        aggregate_stats["total_src"],
        aggregate_stats["evaluated_src"],
        aggregate_stats["skipped_no_corresponding_tgt"],
        aggregate_stats["skipped_no_candidates"],
        aggregate_stats["skipped_gt_not_in_candidates"],
        aggregate_stats["filtered_too_few_candidates"],
        aggregate_stats["pred_missing"],
    )

    entry = build_result_entry(
        "geometric_soco",
        "default",
        model,
        output_dir,
        cfg,
        {"pck": mean_pck},
        dataset="GeometricSOCO",
        stats=aggregate_stats,
    )
    append_jsonl(resolve_results_path(cfg, "correspondence_geometric_soco.jsonl"), entry)
