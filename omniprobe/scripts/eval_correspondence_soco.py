"""Evaluate semantic & concept correspondence on Object_Correspondence pairs."""


from datetime import datetime
import os
from pathlib import Path
import random
import json
import pickle
from collections.abc import Sequence
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from omniprobe.datasets.soco import SOCODataset
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.utils.correspondence import argmax_2d, soft_argmax_2d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_square_patch_size(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return None
        first = int(value[0])
        if len(value) > 1 and int(value[1]) != first:
            raise ValueError(f"Non-square patch size is not supported: {value}.")
        return first
    return int(value)


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _feature_grid_hw(model, image_size: int) -> tuple[int, int]:
    device = _get_model_device(model)
    images = torch.randn(1, 3, image_size, image_size, device=device)
    feats = model(images)
    if isinstance(feats, list):
        if len(feats) == 0:
            raise ValueError("Model returned an empty feature list.")
        feats = feats[0]
    if feats.ndim != 4:
        raise ValueError(f"Expected 4D dense features, got shape {tuple(feats.shape)}")
    return int(feats.shape[-2]), int(feats.shape[-1])


def _infer_patch_size_from_forward(model, probe_image_size: int) -> int:
    feat_h, feat_w = _feature_grid_hw(model, probe_image_size)
    if feat_h != feat_w:
        raise ValueError(
            f"Expected square feature map for square input, got {(feat_h, feat_w)}"
        )
    if feat_h <= 0:
        raise ValueError(f"Invalid feature map size: {(feat_h, feat_w)}")
    inferred = int(round(probe_image_size / feat_h))
    if inferred <= 0:
        raise ValueError(
            f"Failed to infer patch size from input {probe_image_size} and grid {feat_h}"
        )
    return inferred


def _resolve_patch_size(model, probe_image_size: int) -> tuple[int, str]:
    patch_size = _to_square_patch_size(getattr(model, "patch_size", None))
    if patch_size is not None and patch_size > 0:
        return patch_size, "model.patch_size"
    inferred = _infer_patch_size_from_forward(model, probe_image_size)
    return inferred, "inferred_from_forward"


def _resolve_effective_image_size(cfg: DictConfig, model) -> Dict[str, Any]:
    image_size = int(cfg.image_size)
    num_patches = int(cfg.get("num_patches", 60))
    fixed_patched_size = bool(cfg.get("fixed_patched_size", False))
    patch_size, patch_size_source = _resolve_patch_size(model, probe_image_size=image_size)

    if patch_size <= 0:
        raise ValueError(f"Resolved patch_size must be > 0, got {patch_size}")

    if not fixed_patched_size:
        return {
            "image_size": image_size,
            "patch_size": patch_size,
            "patch_size_source": patch_size_source,
            "verified_grid_hw": None,
        }

    if num_patches <= 0:
        raise ValueError(f"num_patches must be > 0, got {num_patches}")

    effective_image_size = num_patches * patch_size
    feat_h, feat_w = _feature_grid_hw(model, effective_image_size)
    if feat_h != num_patches or feat_w != num_patches:
        raise ValueError(
            "fixed_patched_size=True requested "
            f"{num_patches}x{num_patches} patches, but got {feat_h}x{feat_w} "
            f"for image_size={effective_image_size} and patch_size={patch_size} "
            f"({patch_size_source})."
        )
    return {
        "image_size": effective_image_size,
        "patch_size": patch_size,
        "patch_size_source": patch_size_source,
        "verified_grid_hw": (feat_h, feat_w),
    }


def compute_predictions(
    model,
    instance,
    mask_feats=False,
    return_feats=False,
    soft_eval=False,
    soft_eval_beta=0.02,
    soft_eval_window=7,
):
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, meta = instance

    device = _get_model_device(model)
    images = torch.stack((img_i, img_j)).to(device)
    masks = torch.stack((mask_i, mask_j)).to(device)
    masks = torch.nn.functional.avg_pool2d(masks.float(), 16)
    masks = masks > 4 / (16 ** 2)

    feats = model(images)
    if isinstance(feats, list):
        feats = torch.cat(feats, dim=1)
    feats = nn_F.normalize(feats, p=2, dim=1)
    if mask_feats:
        feats = feats * masks

    feats_i = feats[0]
    feats_j = feats[1]

    kps_i = kps_i.float()
    kps_j = kps_j.float()
    size = images.shape[-1]
    kps_i[:, :2] = kps_i[:, :2] / size
    kps_j[:, :2] = kps_j[:, :2] / size

    kps_i_ndc = (kps_i[:, :2] * 2 - 1)[None, None].to(device)
    kp_i_F = nn_F.grid_sample(
        feats_i[None, :], kps_i_ndc, mode="bilinear", align_corners=True
    )
    kp_i_F = kp_i_F[0, :, 0].t()
    heatmaps = torch.einsum("kf,fhw->khw", kp_i_F, feats_j)
    if soft_eval:
        pred_kp = (
            soft_argmax_2d(heatmaps, beta=soft_eval_beta, window=soft_eval_window)
            .float()
            .cpu()
            / feats.shape[-1]
        )
    else:
        pred_kp = argmax_2d(heatmaps, max_value=True).float().cpu() / feats.shape[-1]

    output = {
        "pred": pred_kp,
        "gt_src": kps_i,
        "gt_trg": kps_j,
        "thresh_scale": thresh_scale,
        "meta": meta,
    }
    if return_feats:
        output["images"] = images.detach().cpu()
        output["heatmaps"] = heatmaps.detach().cpu()

    return output


def evaluate_matches(pred_output, thresh):
    pred = pred_output["pred"]
    kps_i = pred_output["gt_src"]
    kps_j = pred_output["gt_trg"]
    scale = pred_output["thresh_scale"]
    meta = pred_output["meta"]

    sem_errors = []
    for src_idx, tgt_idx in meta["semantic_pairs"]:
        if kps_i[src_idx, 2] == 0 or kps_j[tgt_idx, 2] == 0:
            continue
        err = (pred[src_idx] - kps_j[tgt_idx, :2]).norm(p=2)
        sem_errors.append(err / scale)

    sem_errors = torch.tensor(sem_errors) if sem_errors else torch.tensor([])

    concept_errors = []
    for src_idx, tgt_indices in meta["concept_map"].items():
        if kps_i[src_idx, 2] == 0:
            continue
        candidate = []
        for tgt_idx in tgt_indices:
            if kps_j[tgt_idx, 2] == 0:
                continue
            err = (pred[src_idx] - kps_j[tgt_idx, :2]).norm(p=2)
            candidate.append(err)
        if candidate:
            concept_errors.append(min(candidate) / scale)

    concept_errors = torch.tensor(concept_errors) if concept_errors else torch.tensor([])

    metrics = {
        "semantic": sem_errors,
        "concept": concept_errors,
        "meta": meta,
    }
    return metrics


def _serialize_pred_output(pred_output: Dict[str, Any], class_name: Optional[str] = None) -> Dict[str, Any]:
    """Convert pred_output dict (with tensors) to JSON-serializable structure."""
    out = {}
    for k, v in pred_output.items():
        if isinstance(v, torch.Tensor):
            # Convert small tensors to list; large feature maps aren't logged (we never pass feats here)
            out[k] = v.detach().cpu().tolist()
        else:
            out[k] = v
    if class_name is not None:
        out["class_name"] = class_name
    return out


def evaluate_dataset(
    model,
    dataset,
    thresh,
    mask_feats=False,
    log_fh=None,
    class_name: Optional[str] = None,
    record_list: Optional[list] = None,
    soft_eval=False,
    soft_eval_beta=0.02,
    soft_eval_window=7,
    reduction: str = "pair",
):
    """Evaluate dataset and optionally log per-pair predictions.

    Args:
        model: backbone producing dense features.
        dataset: SOCODataset instance.
        thresh: recall threshold.
        mask_feats: whether to mask features by foreground mask.
        log_fh: file handle for writing JSON lines of pred_output (optional).
        class_name: name of the class/category (optional, added to log).
        record_list: list that collects serialized predictions for optional pickle dump.
    Returns:
        (semantic_recall, concept_recall)
    """
    semantic_all = []
    concept_all = []
    for idx in range(len(dataset)):
        pred_output = compute_predictions(
            model,
            dataset.__getitem__(idx),
            mask_feats,
            soft_eval=soft_eval,
            soft_eval_beta=soft_eval_beta,
            soft_eval_window=soft_eval_window,
        )
        metrics = evaluate_matches(pred_output, thresh)
        if metrics["semantic"].numel() > 0:
            if reduction == "keypoint":
                semantic_all.append(metrics["semantic"])
            else:
                semantic_all.append((metrics["semantic"] < thresh).float().mean())
        if metrics["concept"].numel() > 0:
            if reduction == "keypoint":
                concept_all.append(metrics["concept"])
            else:
                concept_all.append((metrics["concept"] < thresh).float().mean())

        if log_fh is not None or record_list is not None:
            # Remove large tensors if accidentally present
            safe_pred = {k: v for k, v in pred_output.items() if k not in {"images", "heatmaps"}}
            serialized = _serialize_pred_output(safe_pred, class_name)
            if log_fh is not None:
                json_line = json.dumps(serialized)
                log_fh.write(json_line + "\n")
            if record_list is not None:
                record_list.append(serialized)

    def recall_from_errors(errors):
        if not errors:
            return float("nan")
        if reduction == "keypoint":
            errs = torch.cat(errors)
            return (errs < thresh).float().mean().item() * 100.0
        vals = torch.stack([x.float() for x in errors])
        return vals.mean().item() * 100.0

    return recall_from_errors(semantic_all), recall_from_errors(concept_all)


@torch.no_grad()
def visualize_samples(
    model,
    dataset,
    out_dir,
    num_samples=4,
    mask_feats=False,
    soft_eval=False,
    soft_eval_beta=0.02,
    soft_eval_window=7,
):
    os.makedirs(out_dir, exist_ok=True)
    chosen = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    for idx in chosen:
        instance = dataset.__getitem__(idx)
        pred_output = compute_predictions(
            model,
            instance,
            mask_feats=mask_feats,
            return_feats=True,
            soft_eval=soft_eval,
            soft_eval_beta=soft_eval_beta,
            soft_eval_window=soft_eval_window,
        )
        images = pred_output["images"].cpu()
        src_img = images[0].permute(1, 2, 0).numpy()
        trg_img = images[1].permute(1, 2, 0).numpy()
        src_img = (src_img - src_img.min()) / (src_img.max() - src_img.min() + 1e-6)
        trg_img = (trg_img - trg_img.min()) / (trg_img.max() - trg_img.min() + 1e-6)

        pred = pred_output["pred"].detach().numpy()
        gt_src = pred_output["gt_src"].detach().numpy()
        gt_trg = pred_output["gt_trg"].detach().numpy()
        meta = pred_output["meta"]

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_img)
        axes[1].imshow(trg_img)
        axes[0].set_title(f"{meta['src_class']} | {meta['src_name']}")
        axes[1].set_title(f"{meta['trg_class']} | {meta['trg_name']}")
        axes[0].axis("off")
        axes[1].axis("off")

        semantic_src = [s for s, _ in meta["semantic_pairs"]]
        semantic_trg = [t for _, t in meta["semantic_pairs"]]
        semantic_src_set = set(semantic_src)
        semantic_trg_set = set(semantic_trg)

        src_names = meta.get("src_names") or [f"kp_{i}" for i in range(len(gt_src))]
        trg_names = meta.get("trg_names") or [f"kp_{i}" for i in range(len(gt_trg))]
        src_concepts = meta.get("src_concepts") or [None] * len(src_names)
        trg_concepts = meta.get("trg_concepts") or [None] * len(trg_names)

        src_name_to_idx = {name: idx for idx, name in enumerate(src_names)}
        trg_name_to_idx = {name: idx for idx, name in enumerate(trg_names)}

        concept_only_src_idx = set()
        concept_only_trg_idx = set()
        concept_src_label: Dict[int, Optional[str]] = {}
        concept_trg_label: Dict[int, Optional[str]] = {}
        for group in meta.get("concept_matches", []):
            cname = group.get("concept_name")
            for s_name in group.get("src_keypoints", []):
                idx = src_name_to_idx.get(s_name)
                if idx is None or idx in semantic_src_set:
                    continue
                concept_only_src_idx.add(idx)
                if cname:
                    concept_src_label[idx] = cname
            for t_name in group.get("trg_keypoints", []):
                idx = trg_name_to_idx.get(t_name)
                if idx is None or idx in semantic_trg_set:
                    continue
                concept_only_trg_idx.add(idx)
                if cname:
                    concept_trg_label[idx] = cname

        size = images.shape[-1]
        gt_src_pix = gt_src[:, :2] * size
        gt_trg_pix = gt_trg[:, :2] * size
        pred_pix = pred * size

        axes[0].scatter(gt_src_pix[semantic_src, 0], gt_src_pix[semantic_src, 1], c="lime", s=40)
        axes[1].scatter(gt_trg_pix[semantic_trg, 0], gt_trg_pix[semantic_trg, 1], c="lime", s=40, label="semantic GT")

        if concept_only_src_idx:
            idx_list = list(concept_only_src_idx)
            axes[0].scatter(gt_src_pix[idx_list, 0], gt_src_pix[idx_list, 1], c="orange", s=30)
        if concept_only_trg_idx:
            idx_list = list(concept_only_trg_idx)
            axes[1].scatter(
                gt_trg_pix[idx_list, 0],
                gt_trg_pix[idx_list, 1],
                c="orange",
                s=30,
                label="concept-only GT",
            )

        axes[1].scatter(pred_pix[:, 0], pred_pix[:, 1], c="red", s=15, alpha=0.6, label="pred")

        def annotate(ax, points, indices, names, base_concepts, override_concepts, color):
            for idx in indices:
                if idx >= points.shape[0] or idx >= len(names):
                    continue
                label = names[idx]
                concept = override_concepts.get(idx)
                if not concept and idx < len(base_concepts):
                    concept = base_concepts[idx]
                if concept:
                    label = f"{label}\n[{concept}]"
                ax.text(
                    float(points[idx, 0]) + 2,
                    float(points[idx, 1]) + 2,
                    label,
                    color=color,
                    fontsize=6,
                    ha="left",
                    va="bottom",
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1),
                )

        annotate(axes[0], gt_src_pix, semantic_src, src_names, src_concepts, {}, "lime")
        annotate(axes[1], gt_trg_pix, semantic_trg, trg_names, trg_concepts, {}, "lime")
        annotate(
            axes[0],
            gt_src_pix,
            list(concept_only_src_idx),
            src_names,
            src_concepts,
            concept_src_label,
            "orange",
        )
        annotate(
            axes[1],
            gt_trg_pix,
            list(concept_only_trg_idx),
            trg_names,
            trg_concepts,
            concept_trg_label,
            "orange",
        )

        fig.legend(loc="lower center", ncol=3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{meta['pair_group']}_{meta['pair_id']:05d}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def run_task(cfg: DictConfig):
    set_seed(cfg.random_seed)
    output_dir = HydraConfig.get().run.dir
    logger.info(f"Output dir: {output_dir}")

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
    elif isinstance(cfg.classes, ListConfig):
        classes = list(cfg.classes)
    else:
        classes = cfg.classes

    if classes is None:
        classes = all_classes

    thresh = cfg.thresh
    results = {}

    # Open per-run prediction log file once.
    pred_log_path = os.path.join(output_dir, "pred_outputs_soco.json")
    pred_pkl_path = os.path.join(output_dir, "pred_outputs_soco.pkl")
    logger.info(f"Logging per-pair predictions to {pred_log_path}")
    pred_records = []
    with open(pred_log_path, "w") as pred_log_fh:
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
                logger.warning(f"No pairs for class {class_name}")
                continue
            recall_sem, recall_concept = evaluate_dataset(
                model,
                dataset,
                thresh,
                cfg.mask_feats,
                log_fh=pred_log_fh,
                class_name=class_name,
                record_list=pred_records,
                soft_eval=cfg.soft_eval,
                soft_eval_beta=cfg.soft_eval_beta,
                soft_eval_window=cfg.soft_eval_window,
                reduction=cfg.reduction,
            )
            logger.info(
                f"Recall@{thresh:.2f} {class_name:15s} | semantic {recall_sem:6.2f} | concept {recall_concept:6.2f}"
            )
            results[class_name] = (recall_sem, recall_concept)

            if cfg.num_visualize > 0:
                vis_dir = os.path.join(output_dir, "vis", class_name)
                visualize_samples(
                    model,
                    dataset,
                    vis_dir,
                    cfg.num_visualize,
                    cfg.mask_feats,
                    soft_eval=cfg.soft_eval,
                    soft_eval_beta=cfg.soft_eval_beta,
                    soft_eval_window=cfg.soft_eval_window,
                )

    with open(pred_pkl_path, "wb") as pred_pkl_fh:
        pickle.dump(pred_records, pred_pkl_fh)
    logger.info(f"Wrote pickle predictions to {pred_pkl_path}")

    if not results:
        logger.error("No results computed")
        return

    sem_values = [v[0] for v in results.values() if not np.isnan(v[0])]
    concept_values = [v[1] for v in results.values() if not np.isnan(v[1])]
    sem_mean = float(np.mean(sem_values)) if sem_values else float("nan")
    concept_mean = float(np.mean(concept_values)) if concept_values else float("nan")
    logger.info(
        f"Mean Recall@{thresh:.2f} | semantic {sem_mean:6.2f} | concept {concept_mean:6.2f}"
    )

    time = datetime.now().strftime("%d%m%Y-%H%M")
    model_name = None
    try:
        model_name = HydraConfig.get().runtime.choices.get("backbone")
    except Exception:
        model_name = None
    if model_name is None:
        model_name = getattr(model, "checkpoint_name", "unknown")
    exp_info = "; ".join(
        [
            model_name,
            str(getattr(model, "patch_size", "")),
            str(getattr(model, "layer", "")),
            "SOCO",
            # "|".join(classes),
        ]
    )
    mode_name = "soft_argmax" if cfg.soft_eval else "nn"
    entry = build_result_entry(
        "soco",
        mode_name,
        model,
        output_dir,
        cfg,
        {"semantic": sem_mean, "concept": concept_mean},
        dataset="SOCO",
    )
    append_jsonl(resolve_results_path(cfg, "correspondence_soco.jsonl"), entry)
