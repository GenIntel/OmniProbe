"""
Evaluate semantic & concept correspondence on SOCO pairs with linear probe
training.

Uses predefined train/test splits from PairAnnotations/trainsplits/.
"""

from datetime import datetime
import os
from pathlib import Path
import json
import pickle
import random
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_F
import torch.optim as optim
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from omniprobe.datasets.soco import SOCODataset
from omniprobe.utils.correspondence import argmax_2d
from omniprobe.models.correspondence_probe import build_correspondence_probe
from omniprobe.runtime import append_jsonl, artifact_dir, build_result_entry, resolve_results_path
from omniprobe.utils.eval_helpers import (
    correspondence_image_size_result_fields,
    log_correspondence_image_size,
    resolve_correspondence_image_size,
)
from omniprobe.utils.progress import progress


# ========== Seed & Utilities ==========


def _get_model_device(model):
    return next(model.parameters()).device

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def soco_collate_fn(batch):
    """
    Custom collate function for SOCODataset.
    
    Each sample is: (img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, meta)
    """
    img_i_list = []
    mask_i_list = []
    kps_i_list = []
    img_j_list = []
    mask_j_list = []
    kps_j_list = []
    thresh_scale_list = []
    meta_list = []
    
    for sample in batch:
        img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, meta = sample
        img_i_list.append(img_i)
        mask_i_list.append(mask_i)
        kps_i_list.append(kps_i)
        img_j_list.append(img_j)
        mask_j_list.append(mask_j)
        kps_j_list.append(kps_j)
        thresh_scale_list.append(thresh_scale)
        meta_list.append(meta)
    
    # Stack images into batch tensors
    img_i_batch = torch.stack(img_i_list, dim=0)
    img_j_batch = torch.stack(img_j_list, dim=0)
    mask_i_batch = torch.stack(mask_i_list, dim=0)
    mask_j_batch = torch.stack(mask_j_list, dim=0)
    
    return {
        'img_i': img_i_batch,
        'img_j': img_j_batch,
        'mask_i': mask_i_batch,
        'mask_j': mask_j_batch,
        'kps_i': kps_i_list,
        'kps_j': kps_j_list,
        'thresh_scale': thresh_scale_list,
        'meta': meta_list,
    }


# ========== Feature Extraction ==========

def extract_features(model, images, probe=None):
    """Extract features from backbone and optionally transform with probe."""
    feats = model(images)
    if isinstance(feats, list):
        feats = torch.cat(feats, dim=1)
    
    if probe is not None:
        feats = probe(feats)
    
    feats = nn_F.normalize(feats, p=2, dim=1)
    return feats


# ========== Prediction & Evaluation ==========

def compute_predictions_with_probe(model, probe, instance, mask_feats=False, return_feats=False):
    """Compute predictions using backbone + optional linear probe."""
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, meta = instance
    device = _get_model_device(model)

    images = torch.stack((img_i, img_j)).to(device)
    masks = torch.stack((mask_i, mask_j)).to(device)

    feats = extract_features(model, images, probe)
    
    # Resize masks to match feature map size
    if mask_feats:
        feat_h, feat_w = feats.shape[-2:]
        masks = nn_F.interpolate(masks, size=(feat_h, feat_w), mode='nearest')
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
    """Evaluate semantic and concept matches."""
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

    return {
        "semantic": sem_errors,
        "concept": concept_errors,
        "meta": meta,
    }


def _serialize_pred_output(pred_output: dict[str, Any], class_name: Optional[str] = None) -> dict[str, Any]:
    """Convert pred_output dict to JSON-serializable structure."""
    out = {}
    for k, v in pred_output.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().tolist()
        else:
            out[k] = v
    if class_name is not None:
        out["class_name"] = class_name
    return out


def evaluate_dataset_with_probe(
    model,
    probe,
    dataset,
    thresh,
    mask_feats=False,
    verbose=False,
    log_fh=None,
    class_name: Optional[str] = None,
    record_list: Optional[list] = None,
):
    """Evaluate dataset using backbone + linear probe."""
    if probe is not None:
        probe.eval()
    
    iterator = (
        progress(range(len(dataset)), desc="SOCO linear-probe evaluation")
        if verbose
        else range(len(dataset))
    )
    semantic_all = []
    concept_all = []

    for idx in iterator:
        with torch.no_grad():
            pred_output = compute_predictions_with_probe(
                model, probe, dataset.__getitem__(idx), mask_feats
            )
        metrics = evaluate_matches(pred_output, thresh)
        if metrics["semantic"].numel() > 0:
            semantic_all.append(metrics["semantic"])
        if metrics["concept"].numel() > 0:
            concept_all.append(metrics["concept"])

        if log_fh is not None or record_list is not None:
            safe_pred = {k: v for k, v in pred_output.items() if k not in {"images", "heatmaps"}}
            serialized = _serialize_pred_output(safe_pred, class_name)
            if log_fh is not None:
                log_fh.write(json.dumps(serialized) + "\n")
            if record_list is not None:
                record_list.append(serialized)

    def recall_from_errors(errors):
        if not errors:
            return float("nan")
        errs = torch.cat(errors)
        return (errs < thresh).float().mean().item() * 100.0

    return recall_from_errors(semantic_all), recall_from_errors(concept_all)


# ========== Training ==========

def compute_batch_training_loss(feats_i, feats_j, kps_i_list, kps_j_list, meta_list, image_size, temperature=0.07):
    """
    Compute contrastive loss for a batch of image pairs.

    Uses semantic pairs from meta for supervision. Each source keypoint is
    contrasted against all H*W target spatial locations.
    """
    batch_size = feats_i.shape[0]
    device = feats_i.device
    total_loss = 0.0
    valid_pairs = 0

    for b in range(batch_size):
        # Clone to avoid modifying original tensors
        kps_i = kps_i_list[b].float().clone()
        kps_j = kps_j_list[b].float().clone()
        meta = meta_list[b]

        # Normalize keypoints to [0, 1]
        kps_i[:, :2] = kps_i[:, :2] / image_size
        kps_j[:, :2] = kps_j[:, :2] / image_size

        # Use semantic pairs for supervision
        semantic_pairs = meta.get("semantic_pairs", [])
        if len(semantic_pairs) == 0:
            continue

        # Extract valid semantic pair indices
        valid_src_indices = []
        valid_tgt_indices = []
        for src_idx, tgt_idx in semantic_pairs:
            if kps_i[src_idx, 2] == 1 and kps_j[tgt_idx, 2] == 1:
                valid_src_indices.append(src_idx)
                valid_tgt_indices.append(tgt_idx)

        if len(valid_src_indices) == 0:
            continue

        valid_src_indices = torch.tensor(valid_src_indices)
        valid_tgt_indices = torch.tensor(valid_tgt_indices)

        # Get source keypoint features
        valid_kps_i = kps_i[valid_src_indices]
        kps_i_ndc = (valid_kps_i[:, :2].float() * 2 - 1)[None, None].to(device)
        kp_i_F = nn_F.grid_sample(
            feats_i[b:b+1], kps_i_ndc, mode="bilinear", align_corners=True
        )
        kp_i_F = kp_i_F[0, :, 0].t()  # (K, C)

        valid_kps_j = kps_j[valid_tgt_indices]

        # Contrast against ALL target spatial locations (K × H*W)
        C, H, W = feats_j[b].shape
        feats_j_flat = feats_j[b].reshape(C, H * W).t()  # (H*W, C)
        sim_all = torch.mm(kp_i_F, feats_j_flat.t()) / temperature  # (K, H*W)

        # Labels: nearest grid cell to each GT target keypoint
        target_col = (valid_kps_j[:, 0] * (W - 1)).round().long().to(device)
        target_row = (valid_kps_j[:, 1] * (H - 1)).round().long().to(device)
        target_col = target_col.clamp(0, W - 1)
        target_row = target_row.clamp(0, H - 1)
        labels = target_row * W + target_col

        loss = nn_F.cross_entropy(sim_all, labels)
        
        total_loss += loss
        valid_pairs += 1
    
    if valid_pairs == 0:
        return None
    
    return total_loss / valid_pairs


def train_probe_epoch(model, probe, train_loader, optimizer, cfg, epoch, image_size, device):
    """Train the linear probe for one epoch with batched data."""
    probe.train()
    
    loss_meter = AverageMeter()
    
    pbar = progress(train_loader, desc=f"Epoch {epoch}")
    
    num_batches_with_loss = 0
    num_batches_skipped = 0
    
    for batch in pbar:
        img_i = batch["img_i"].to(device)
        img_j = batch["img_j"].to(device)
        kps_i_list = batch['kps_i']
        kps_j_list = batch['kps_j']
        meta_list = batch['meta']
        
        optimizer.zero_grad()
        
        # Stack source and target images for efficient backbone forward pass
        batch_size = img_i.shape[0]
        images = torch.cat([img_i, img_j], dim=0)
        
        with torch.no_grad():
            feats = model(images)
            if isinstance(feats, list):
                feats = torch.cat(feats, dim=1)
        
        # Apply probe (trainable)
        feats = probe(feats)
        feats = nn_F.normalize(feats, p=2, dim=1)
        
        # Split back into source and target features
        feats_i = feats[:batch_size]
        feats_j = feats[batch_size:]
        
        # Compute batch loss
        loss = compute_batch_training_loss(
            feats_i, feats_j, kps_i_list, kps_j_list, meta_list,
            image_size, temperature=cfg.temperature,
        )
        
        if loss is not None:
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), batch_size)
            num_batches_with_loss += 1
        else:
            num_batches_skipped += 1
        
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    
    if num_batches_skipped > 0:
        logger.warning(f"Epoch {epoch}: {num_batches_skipped} batches skipped (no valid semantic pairs)")
    if num_batches_with_loss == 0:
        logger.warning(f"Epoch {epoch}: No batches had valid semantic pairs for training!")
    
    return loss_meter.avg


def get_feature_dim(model, image_size, device):
    """Get the feature dimension from the backbone."""
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
        feats = model(dummy_input)
        if isinstance(feats, list):
            feats = torch.cat(feats, dim=1)
        return feats.shape[1]


def run_task(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir
    logger.info(f"Output dir: {output_dir}")
    
    # Set seeds for reproducibility
    seed = cfg.get("random_seed", 8)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    device = torch.device(str(cfg.device)) if "device" in cfg else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ===== Get model =====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to(device)
    model.eval()
    image_size_info = resolve_correspondence_image_size(cfg, model)
    effective_image_size = int(image_size_info["effective_image_size"])
    log_correspondence_image_size(image_size_info)

    # Get feature dimension
    feat_dim = get_feature_dim(model, effective_image_size, device)
    logger.info(f"Backbone feature dimension: {feat_dim}")

    # ===== Build linear probe =====
    probe_output_dim = cfg.probe.get("output_dim", feat_dim)
    probe_hidden_dim = cfg.probe.get("hidden_dim", feat_dim)
    probe_init_mode = cfg.probe.get("init_mode", "identity")
    
    probe = build_correspondence_probe(
        probe_type=cfg.probe.type,
        input_dim=feat_dim,
        output_dim=probe_output_dim,
        hidden_dim=probe_hidden_dim,
        bias=cfg.probe.get("bias", True),
        init_mode=probe_init_mode,
    )
    probe = probe.to(device)
    
    # Load pre-trained probe if checkpoint path is provided
    probe_checkpoint = cfg.get("probe_checkpoint", None)
    if probe_checkpoint is not None and os.path.exists(probe_checkpoint):
        logger.info(f"Loading pre-trained probe from {probe_checkpoint}")
        probe.load_state_dict(torch.load(probe_checkpoint, map_location=device))
    
    num_params = sum(p.numel() for p in probe.parameters())
    logger.info(f"Linear probe type: {cfg.probe.type}, init_mode: {probe_init_mode}")
    logger.info(f"Linear probe parameters: {num_params}")

    # ===== Resolve class list =====
    data_root = cfg.data_root
    test_pair_subdir = cfg.test_pair_subdir
    test_pair_root = Path(test_pair_subdir)
    if not test_pair_root.is_absolute():
        test_pair_root = Path(data_root) / test_pair_subdir
    all_classes = sorted([d.name for d in test_pair_root.iterdir() if d.is_dir()])

    if isinstance(cfg.classes, str) and cfg.classes == "all":
        classes = all_classes
    elif isinstance(cfg.classes, ListConfig):
        classes = list(cfg.classes)
    else:
        classes = cfg.classes if cfg.classes else all_classes

    thresh = cfg.thresh

    # ===== Load train and test datasets from predefined splits =====
    dataset_kwargs = dict(
        root=data_root,
        image_size=effective_image_size,
        image_mean=cfg.image_mean,
        use_bbox=cfg.use_bbox,
        max_pairs=cfg.max_pairs,
    )
    train_dataset = SOCODataset(pair_subdir=cfg.train_pair_subdir, **dataset_kwargs)
    test_dataset = SOCODataset(pair_subdir=test_pair_subdir, **dataset_kwargs)
    logger.info(f"Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}")

    # ===== Evaluate with initial probe (before training) =====
    if cfg.get("eval_before_training", True):
        logger.info("=" * 50)
        logger.info("Evaluating with initial probe (before training)...")
        logger.info("=" * 50)
        
        recall_sem, recall_concept = evaluate_dataset_with_probe(
            model, probe, test_dataset, thresh, cfg.mask_feats, verbose=True
        )
        logger.info(f"Before training - Semantic: {recall_sem:.2f}, Concept: {recall_concept:.2f}")

    # ===== Train linear probe =====
    if cfg.train.enabled:
        logger.info("=" * 50)
        logger.info("Training linear probe...")
        logger.info("=" * 50)
        
        # Build DataLoader
        batch_size = cfg.train.get("batch_size", 8)
        num_workers = cfg.train.get("num_workers", 4)
        
        # Adjust batch size if larger than dataset
        actual_batch_size = min(batch_size, len(train_dataset))
        if actual_batch_size < batch_size:
            logger.warning(f"Reducing batch_size from {batch_size} to {actual_batch_size} (dataset has only {len(train_dataset)} samples)")
        
        # Only drop_last if we have more samples than batch_size
        drop_last = len(train_dataset) > actual_batch_size
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=soco_collate_fn,
            pin_memory=True,
            drop_last=drop_last,
            generator=g,
        )
        logger.info(f"Training with batch_size={actual_batch_size}, num_workers={num_workers}, drop_last={drop_last}")
        
        # Build optimizer
        optimizer = optim.Adam(
            probe.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.get("weight_decay", 0.0),
        )
        
        # Optional: learning rate scheduler
        if cfg.train.get("use_scheduler", False):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.01
            )
        else:
            scheduler = None
        
        best_recall = 0
        best_probe_state = None
        
        for epoch in range(1, cfg.train.epochs + 1):
            set_seed(seed + epoch)
            
            avg_loss = train_probe_epoch(
                model, probe, train_loader, optimizer, cfg.train, epoch, effective_image_size, device
            )
            logger.info(f"Epoch {epoch}/{cfg.train.epochs} - Loss: {avg_loss:.4f}")
            
            if scheduler is not None:
                scheduler.step()
            
            # Evaluate every N epochs
            if epoch % cfg.train.get("eval_every", 1) == 0 or epoch == cfg.train.epochs:
                recall_sem, recall_concept = evaluate_dataset_with_probe(
                    model, probe, test_dataset, thresh, cfg.mask_feats, verbose=False
                )
                # Use average of semantic and concept as validation metric
                avg_recall = (recall_sem + recall_concept) / 2 if not (np.isnan(recall_sem) or np.isnan(recall_concept)) else recall_sem
                logger.info(f"Epoch {epoch} - Test Semantic: {recall_sem:.2f}, Concept: {recall_concept:.2f}")
                
                if avg_recall > best_recall:
                    best_recall = avg_recall
                    best_probe_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                    logger.info(f"New best avg recall: {best_recall:.2f}")
        
        # Load best model
        if best_probe_state is not None:
            probe.load_state_dict(best_probe_state)
            logger.info(f"Loaded best probe with avg recall: {best_recall:.2f}")
        
        # Save trained probe
        probe_save_path = artifact_dir(cfg, "checkpoints") / "trained_probe.pth"
        torch.save(probe.state_dict(), probe_save_path)
        logger.info(f"Saved trained probe to {probe_save_path}")

    # ===== Final evaluation on test set per class =====
    logger.info("=" * 50)
    logger.info("Final evaluation on test split (per class)...")
    logger.info("=" * 50)
    
    pred_dir = artifact_dir(cfg, "predictions")
    pred_log_path = pred_dir / "pred_outputs_soco_linear_probe.json"
    pred_pkl_path = pred_dir / "pred_outputs_soco_linear_probe.pkl"
    logger.info(f"Logging per-pair predictions to {pred_log_path}")

    results = {}
    pred_records = []

    with open(pred_log_path, "w") as pred_log_fh:
        for class_name in classes:
            class_test_dataset = SOCODataset(
                pair_subdir=test_pair_subdir, class_name=class_name, **dataset_kwargs
            )
            if len(class_test_dataset) == 0:
                logger.warning(f"No test pairs for class {class_name}")
                continue

            recall_sem, recall_concept = evaluate_dataset_with_probe(
                model,
                probe,
                class_test_dataset,
                thresh,
                cfg.mask_feats,
                log_fh=pred_log_fh,
                class_name=class_name,
                record_list=pred_records,
            )
            logger.info(
                f"Recall@{thresh:.2f} {class_name:15s} | semantic {recall_sem:6.2f} | concept {recall_concept:6.2f}"
            )
            results[class_name] = (recall_sem, recall_concept)

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

    # Result summary
    time = datetime.now().strftime("%d%m%Y-%H%M")
    probe_info = f"probe_{cfg.probe.type}"
    if cfg.train.enabled:
        probe_info += f"_trained_ep{cfg.train.epochs}_lr{cfg.train.lr}"
    
    exp_info = ", ".join(
        [
            getattr(model, "checkpoint_name", "unknown"),
            str(getattr(model, "patch_size", "")),
            str(getattr(model, "layer", "")),
            "SOCO",
            probe_info,
        ]
    )
    entry = build_result_entry(
        "soco",
        model,
        output_dir,
        cfg,
        {"semantic": sem_mean, "concept": concept_mean},
        dataset="SOCO",
        **correspondence_image_size_result_fields(image_size_info),
    )
    append_jsonl(resolve_results_path(cfg, "correspondence_soco_linear_probe.jsonl"), entry)
