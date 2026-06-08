from datetime import datetime
import json
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as nn_F
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import einsum
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from omniprobe.datasets.spair import CLASS_IDS, SPairDataset
from omniprobe.runtime import append_jsonl, artifact_dir, build_result_entry, resolve_results_path
from omniprobe.utils.correspondence import argmax_2d
from omniprobe.models.correspondence_probe import build_correspondence_probe
from omniprobe.utils.paths import cfg_or_env_path
from omniprobe.utils.progress import progress

from hydra.core.hydra_config import HydraConfig
import os


# ========== Helper ==========


def _get_model_device(model):
    return next(model.parameters()).device

def to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def _serialize_pred_output(pred_output):
    """Convert tensors in pred_output to Python lists for JSON dumps."""
    out = {}
    for key, value in pred_output.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().tolist()
        else:
            out[key] = value
    return out


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


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: enable for full determinism (may slow training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def spair_collate_fn(batch):
    """
    Custom collate function for SPair dataset.
    
    Each sample is a tuple: (img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, class_name)
    We batch images together but keep keypoints as lists since they have variable length.
    """
    img_i_list = []
    mask_i_list = []
    kps_i_list = []
    img_j_list = []
    mask_j_list = []
    kps_j_list = []
    thresh_scale_list = []
    class_name_list = []
    
    for sample in batch:
        img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, class_name = sample
        img_i_list.append(img_i)
        mask_i_list.append(torch.tensor(np.array(mask_i, dtype=np.float32)))
        kps_i_list.append(kps_i)
        img_j_list.append(img_j)
        mask_j_list.append(torch.tensor(np.array(mask_j, dtype=np.float32)))
        kps_j_list.append(kps_j)
        thresh_scale_list.append(thresh_scale)
        class_name_list.append(class_name)
    
    # Stack images into batch tensors
    img_i_batch = torch.stack(img_i_list, dim=0)  # (B, 3, H, W)
    img_j_batch = torch.stack(img_j_list, dim=0)  # (B, 3, H, W)
    mask_i_batch = torch.stack(mask_i_list, dim=0)  # (B, H, W)
    mask_j_batch = torch.stack(mask_j_list, dim=0)  # (B, H, W)
    
    return {
        'img_i': img_i_batch,
        'img_j': img_j_batch,
        'mask_i': mask_i_batch,
        'mask_j': mask_j_batch,
        'kps_i': kps_i_list,  # List of (K_i, 3) tensors
        'kps_j': kps_j_list,  # List of (K_j, 3) tensors
        'thresh_scale': thresh_scale_list,
        'class_name': class_name_list,
    }


def extract_features(model, images, probe=None):
    """Extract features from backbone and optionally transform with probe."""
    feats = model(images)
    if isinstance(feats, list):
        feats = torch.cat(feats, dim=1)
    
    # Apply probe if provided
    if probe is not None:
        feats = probe(feats)
    
    feats = nn_F.normalize(feats, p=2, dim=1)
    return feats


def compute_predictions_with_probe(model, probe, instance, mask_feats=False, return_heatmaps=False):
    """Compute predictions using backbone + optional linear probe."""
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, class_name = instance
    mask_i = torch.tensor(np.array(mask_i, dtype=float))
    mask_j = torch.tensor(np.array(mask_j, dtype=float))
    device = _get_model_device(model)

    images = torch.stack((img_i, img_j)).to(device)
    masks = torch.stack((mask_i, mask_j)).to(device)
    masks = torch.nn.functional.avg_pool2d(masks.float(), 16)
    masks = masks > 4 / (16 ** 2)

    feats = extract_features(model, images, probe)

    if mask_feats:
        feats = feats * masks

    feats_i = feats[0]
    feats_j = feats[1]

    # normalize kps to [0, 1]
    assert images.shape[-1] == images.shape[-2], "assuming square images here"
    kps_i = kps_i.float()
    kps_j = kps_j.float()
    kps_i[:, :2] = kps_i[:, :2] / images.shape[-1]
    kps_j[:, :2] = kps_j[:, :2] / images.shape[-1]

    # get correspondences
    kps_i_ndc = (kps_i[:, :2].float() * 2 - 1)[None, None].to(device)
    kp_i_F = nn_F.grid_sample(
        feats_i[None, :], kps_i_ndc, mode="bilinear", align_corners=True
    )
    kp_i_F = kp_i_F[0, :, 0].t()

    # get max index in [0,1] range
    heatmaps = einsum(kp_i_F, feats_j, "k f, f h w -> k h w")
    pred_kp = argmax_2d(heatmaps, max_value=True).float().cpu() / feats.shape[-1]

    pred_output = {
        "pred": pred_kp,
        "gt_src": kps_i.detach().cpu(),
        "gt_trg": kps_j.detach().cpu(),
        "thresh_scale": float(thresh_scale),
        "meta": {"class_name": class_name},
    }

    if return_heatmaps:
        pred_output["heatmaps"] = heatmaps.detach().cpu()

    return pred_output


def compute_errors_with_probe(model, probe, instance, mask_feats=False, return_heatmaps=False):
    """Compute errors using backbone + optional linear probe."""
    pred_output = compute_predictions_with_probe(model, probe, instance, mask_feats, return_heatmaps)
    pred_kp = pred_output["pred"]
    kps_i = pred_output["gt_src"]
    kps_j = pred_output["gt_trg"]
    thresh_scale = pred_output["thresh_scale"]

    # compute error and scale to threshold (for all pairs)
    errors = (pred_kp[:, None, :] - kps_j[None, :, :2]).norm(p=2, dim=-1)
    errors = errors / thresh_scale

    # only retain keypoints in both (for now)
    valid_kps = (kps_i[:, None, 2] * kps_j[None, :, 2]) == 1
    in_both = valid_kps.diagonal()

    # max error should be 1, so this excludes invalid from NN-search
    errors[valid_kps.logical_not()] = 1e3

    error_same = errors.diagonal()[in_both]
    error_nn, index_nn = errors[in_both].min(dim=1)
    index_same = in_both.nonzero().squeeze(1)

    return error_same, error_nn, index_same, index_nn, pred_output


def compute_training_loss(model, probe, instance, temperature=0.07):
    """
    Compute contrastive loss for training the linear probe.
    
    Uses InfoNCE loss: for each keypoint in image A, the corresponding
    keypoint location in image B should have the highest similarity.
    """
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, class_name = instance

    device = _get_model_device(model)
    images = torch.stack((img_i, img_j)).to(device)

    feats = extract_features(model, images, probe)

    feats_i = feats[0]  # (C, H, W)
    feats_j = feats[1]  # (C, H, W)

    # normalize kps to [0, 1]
    assert images.shape[-1] == images.shape[-2], "assuming square images here"
    kps_i = kps_i.float()
    kps_j = kps_j.float()
    kps_i[:, :2] = kps_i[:, :2] / images.shape[-1]
    kps_j[:, :2] = kps_j[:, :2] / images.shape[-1]

    # Find valid keypoints (present in both images)
    valid_mask = (kps_i[:, 2] * kps_j[:, 2]) == 1
    if valid_mask.sum() == 0:
        return None  # No valid keypoints for this pair

    valid_kps_i = kps_i[valid_mask]
    valid_kps_j = kps_j[valid_mask]

    # Get source keypoint features
    kps_i_ndc = (valid_kps_i[:, :2].float() * 2 - 1)[None, None].to(device)
    kp_i_F = nn_F.grid_sample(
        feats_i[None, :], kps_i_ndc, mode="bilinear", align_corners=True
    )
    kp_i_F = kp_i_F[0, :, 0].t()  # (K, C)

    # Get target keypoint features (ground truth locations)
    kps_j_ndc = (valid_kps_j[:, :2].float() * 2 - 1)[None, None].to(device)
    kp_j_F = nn_F.grid_sample(
        feats_j[None, :], kps_j_ndc, mode="bilinear", align_corners=True
    )
    kp_j_F = kp_j_F[0, :, 0].t()  # (K, C)

    # Compute similarity matrix
    # (K, C) @ (C, K) = (K, K)
    sim_matrix = torch.mm(kp_i_F, kp_j_F.t()) / temperature

    # InfoNCE loss: diagonal should be the positive pairs
    K = sim_matrix.shape[0]
    labels = torch.arange(K, device=sim_matrix.device)
    
    # Cross-entropy loss (each row: source kp should match target kp at same index)
    loss = nn_F.cross_entropy(sim_matrix, labels)

    return loss


def compute_batch_training_loss(feats_i, feats_j, kps_i_list, kps_j_list, image_size, temperature=0.07):
    """
    Compute contrastive loss for a batch of image pairs.

    Each source keypoint is contrasted against all H*W target spatial locations.
    """
    batch_size = feats_i.shape[0]
    device = feats_i.device
    total_loss = 0.0
    valid_pairs = 0

    for b in range(batch_size):
        kps_i = kps_i_list[b].float()
        kps_j = kps_j_list[b].float()

        # Normalize keypoints to [0, 1]
        kps_i[:, :2] = kps_i[:, :2] / image_size
        kps_j[:, :2] = kps_j[:, :2] / image_size

        # Find valid keypoints (present in both images)
        valid_mask = (kps_i[:, 2] * kps_j[:, 2]) == 1
        if valid_mask.sum() == 0:
            continue

        valid_kps_i = kps_i[valid_mask]
        valid_kps_j = kps_j[valid_mask]

        # Get source keypoint features
        kps_i_ndc = (valid_kps_i[:, :2].float() * 2 - 1)[None, None].to(device)
        kp_i_F = nn_F.grid_sample(
            feats_i[b:b+1], kps_i_ndc, mode="bilinear", align_corners=True
        )
        kp_i_F = kp_i_F[0, :, 0].t()  # (K, C)

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
    
    for batch in pbar:
        img_i = batch["img_i"].to(device)  # (B, 3, H, W)
        img_j = batch["img_j"].to(device)  # (B, 3, H, W)
        kps_i_list = batch['kps_i']
        kps_j_list = batch['kps_j']
        
        optimizer.zero_grad()
        
        # Stack source and target images for efficient backbone forward pass
        # Shape: (2*B, 3, H, W)
        batch_size = img_i.shape[0]
        images = torch.cat([img_i, img_j], dim=0)
        
        with torch.no_grad():
            # Extract features from frozen backbone
            feats = model(images)
            if isinstance(feats, list):
                feats = torch.cat(feats, dim=1)
        
        # Apply probe (trainable)
        feats = probe(feats)
        feats = nn_F.normalize(feats, p=2, dim=1)
        
        # Split back into source and target features
        feats_i = feats[:batch_size]  # (B, C, H, W)
        feats_j = feats[batch_size:]  # (B, C, H, W)
        
        # Compute batch loss
        loss = compute_batch_training_loss(
            feats_i, feats_j, kps_i_list, kps_j_list,
            image_size, temperature=cfg.temperature,
        )
        
        if loss is not None:
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), batch_size)
        
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    
    return loss_meter.avg


def evaluate_dataset_with_probe(
    model,
    probe,
    dataset,
    thresh,
    verbose=False,
    log_fh=None,
    class_name=None,
    subset_name=None,
    record_list=None,
):
    """Evaluate dataset using backbone + linear probe."""
    if probe is not None:
        probe.eval()
    
    iterator = (
        progress(range(len(dataset)), desc="SPair linear-probe evaluation")
        if verbose
        else range(len(dataset))
    )
    errors_all = []
    src_all = []
    tgt_all = []

    for idx in iterator:
        with torch.no_grad():
            error_same, _, index_same, index_nn, pred_output = compute_errors_with_probe(
                model, probe, dataset.__getitem__(idx)
            )
        errors_all.append(error_same)
        src_all.append(index_same)
        tgt_all.append(index_nn)

        if log_fh is not None or record_list is not None:
            meta = dict(pred_output.get("meta", {}))
            if class_name:
                meta["class_name"] = class_name
            meta["pair_index"] = int(idx)
            if subset_name is not None:
                meta["subset_view_diff"] = subset_name
            if hasattr(dataset, "instances"):
                pair_info = dataset.instances[idx]
                if isinstance(pair_info, dict):
                    if "filename" in pair_info:
                        meta.setdefault("pair_filename", pair_info["filename"])
                    if "viewpoint_variation" in pair_info:
                        meta["pair_viewpoint_variation"] = pair_info["viewpoint_variation"]
            pred_output["meta"] = meta
            safe_pred = {k: v for k, v in pred_output.items() if k not in {"images", "heatmaps"}}
            serialized = _serialize_pred_output(safe_pred)
            if log_fh is not None:
                log_fh.write(json.dumps(serialized) + "\n")
            if record_list is not None:
                record_list.append(serialized)

    if not errors_all:
        return float("nan"), None

    errors = torch.cat(errors_all)
    src_ind = torch.cat(src_all)
    tgt_ind = torch.cat(tgt_all)

    if src_ind.numel() > 0 and tgt_ind.numel() > 0:
        kp_max = int(max(src_ind.max().item(), tgt_ind.max().item())) + 1
        confusion = torch.zeros((kp_max, kp_max))
        for src, tgt in torch.stack((src_ind, tgt_ind), dim=1):
            confusion[src, tgt] += 1
    else:
        confusion = None

    recall = (errors < thresh).float().mean().item() * 100.0

    return recall, confusion


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
    vis_dir = artifact_dir(cfg, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    seed = cfg.get("random_seed", 8)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    device = torch.device(str(cfg.device)) if "device" in cfg else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    data_root = cfg_or_env_path(cfg, "data_root", "SPAIR_ROOT", "SPair-71k root")
    thresh = 0.10

    # ===== Get model =====
    backbone_kwargs = dict(output="dense")
    if cfg.multilayer:
        backbone_kwargs["return_multilayer"] = True
    
    model = instantiate(cfg.backbone, **backbone_kwargs)
    model = model.to(device)
    model.eval()  # Freeze backbone
    
    # Get feature dimension
    feat_dim = get_feature_dim(model, cfg.image_size, device)
    logger.info(f"Backbone feature dimension: {feat_dim}")

    # ===== Build linear probe =====
    probe_output_dim = cfg.probe.get("output_dim", feat_dim)
    probe_hidden_dim = cfg.probe.get("hidden_dim", feat_dim)
    probe_init_mode = cfg.probe.get("init_mode", "linear")
    
    probe = build_correspondence_probe(
        probe_type=cfg.probe.type,
        input_dim=feat_dim,
        output_dim=probe_output_dim,
        hidden_dim=probe_hidden_dim,
        bias=cfg.probe.get("bias", True),
        init_mode=probe_init_mode,
    )
    probe = probe.to(device)
    
    num_params = sum(p.numel() for p in probe.parameters())
    logger.info(f"Linear probe type: {cfg.probe.type}, init_mode: {probe_init_mode}")
    logger.info(f"Linear probe parameters: {num_params}")

    # ===== GET DATA LOADERS =====
    if cfg.eval_class == "all":
        classes = list(CLASS_IDS.keys())
    else:
        assert cfg.eval_class in CLASS_IDS
        classes = [cfg.eval_class]

    # ===== Evaluate with random initialization (before training) =====
    if cfg.get("eval_before_training", True):
        logger.info("=" * 50)
        logger.info("Evaluating with randomly initialized probe...")
        logger.info("=" * 50)
        
        test_dataset = SPairDataset(
            data_root,
            "test",
            use_bbox=cfg.use_bbox,
            image_size=cfg.image_size,
            image_mean=cfg.image_mean,
            class_name=None,  # All classes
            num_instances=cfg.get("eval_num_instances", 200),
        )
        
        recall_before, _ = evaluate_dataset_with_probe(
            model, probe, test_dataset, thresh, verbose=True
        )
        logger.info(f"Recall@{thresh} before training (random init): {recall_before:.2f}")

    # ===== Train linear probe =====
    if cfg.train.enabled:
        logger.info("=" * 50)
        logger.info("Training linear probe on train split...")
        logger.info("=" * 50)
        
        # Build training dataset
        train_dataset = SPairDataset(
            data_root,
            "train",
            use_bbox=cfg.use_bbox,
            image_size=cfg.image_size,
            image_mean=cfg.image_mean,
            class_name=None if cfg.train.train_all_classes else cfg.eval_class,
            num_instances=cfg.train.get("num_instances", None),
        )
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
        # Build DataLoader with batching
        batch_size = cfg.train.get("batch_size", 8)
        num_workers = cfg.train.get("num_workers", 4)
        
        # Create generator with fixed seed for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(seed)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=spair_collate_fn,
            pin_memory=True,
            drop_last=True,
            generator=g,
        )
        logger.info(f"Training with batch_size={batch_size}, num_workers={num_workers}")
        
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
            # Set epoch-specific seed for reproducibility across runs
            set_seed(seed + epoch)
            
            avg_loss = train_probe_epoch(
                model, probe, train_loader, optimizer, cfg.train, epoch, cfg.image_size, device
            )
            logger.info(f"Epoch {epoch}/{cfg.train.epochs} - Loss: {avg_loss:.4f}")
            
            if scheduler is not None:
                scheduler.step()
            
            # Evaluate every N epochs
            if epoch % cfg.train.get("eval_every", 5) == 0 or epoch == cfg.train.epochs:
                val_dataset = SPairDataset(
                    data_root,
                    "valid",
                    use_bbox=cfg.use_bbox,
                    image_size=cfg.image_size,
                    image_mean=cfg.image_mean,
                    class_name=None,
                    num_instances=cfg.get("eval_num_instances", 200),
                )
                recall_val, _ = evaluate_dataset_with_probe(
                    model, probe, val_dataset, thresh, verbose=False
                )
                logger.info(f"Epoch {epoch} - Validation Recall@{thresh}: {recall_val:.2f}")
                
                if recall_val > best_recall:
                    best_recall = recall_val
                    best_probe_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                    logger.info(f"New best validation recall: {best_recall:.2f}")
        
        # Load best model
        if best_probe_state is not None:
            probe.load_state_dict(best_probe_state)
            logger.info(f"Loaded best probe with validation recall: {best_recall:.2f}")
        
        # Save trained probe
        probe_save_path = artifact_dir(cfg, "checkpoints") / "trained_probe.pth"
        torch.save(probe.state_dict(), probe_save_path)
        logger.info(f"Saved trained probe to {probe_save_path}")

    # ===== Final evaluation on test set =====
    logger.info("=" * 50)
    logger.info("Final evaluation on test split...")
    logger.info("=" * 50)
    
    pred_dir = artifact_dir(cfg, "predictions")
    pred_log_path = pred_dir / "pred_outputs_spair_correspondence_linear_probe.json"
    pred_pkl_path = pred_dir / "pred_outputs_spair_correspondence_linear_probe.pkl"
    logger.info(f"Logging per-pair predictions to {pred_log_path}")

    class_acc = {}
    pred_records = []
    with open(pred_log_path, "w") as pred_log_fh:
        for class_name in classes:
            recall = []
            confusion = []
            for vp_setting in [0, 1, 2, None]:
                dataset = SPairDataset(
                    data_root,
                    cfg.split,
                    use_bbox=cfg.use_bbox,
                    image_size=cfg.image_size,
                    image_mean=cfg.image_mean,
                    class_name=class_name,
                    num_instances=cfg.num_instances,
                    vp_diff=vp_setting,
                )
                vp_label = "all" if vp_setting is None else f"{vp_setting:3d}"
                subset_name = "all" if vp_setting is None else str(vp_setting)
                if len(dataset) > 0:
                    rec_i, conf_i = evaluate_dataset_with_probe(
                        model,
                        probe,
                        dataset,
                        thresh,
                        log_fh=pred_log_fh,
                        class_name=class_name,
                        subset_name=subset_name,
                        record_list=pred_records,
                    )
                    logger.info(
                        f"Recall@{thresh} {class_name:>13s} {vp_label} |  {rec_i:6.2f}"
                    )
                else:
                    logger.info(f"Recall@{thresh} {class_name:>13s} {vp_label} |  N/A")
                    rec_i, conf_i = -1, None
                recall.append(rec_i)
                confusion.append(conf_i)

            result_log = [f"{_rec:5.1f}" if _rec >= 0 else " N/A " for _rec in recall]
            result_log = "   ".join(result_log)
            logger.info(f"Recall@{thresh} {class_name:>13s}     |  {result_log}")
            class_acc[class_name] = (recall, confusion)

    with open(pred_pkl_path, "wb") as pred_pkl_fh:
        pickle.dump(pred_records, pred_pkl_fh)
    logger.info(f"Wrote pickle predictions to {pred_pkl_path}")

    all_recall = [torch.tensor(class_acc[cls][0], dtype=float) for cls in class_acc]
    all_recall = torch.stack(all_recall, dim=0)
    valid_rec = (all_recall >= 0).float()  # invalid is set to -1
    avg_recall = (all_recall * valid_rec).sum(dim=0) / valid_rec.sum(dim=0)

    for i, vp_diff in enumerate(["0", "1", "2", "all"]):
        logger.info(f"Recall@{thresh}  view diff={vp_diff:>3s} |  {avg_recall[i]:6.2f}")

    # result summary
    time = datetime.now().strftime("%d%m%Y-%H%M")
    probe_info = f"probe_{cfg.probe.type}"
    if cfg.train.enabled:
        probe_info += f"_trained_ep{cfg.train.epochs}_lr{cfg.train.lr}"
    num_instances_label = "all" if cfg.num_instances is None else f"{int(cfg.num_instances):5d}"
    num_instances_value = None if cfg.num_instances is None else int(cfg.num_instances)
    
    exp_info = ", ".join(
        [
            f"{model.checkpoint_name:30s}",
            f"{model.patch_size:2d}",
            f"{str(model.layer):5s}",
            f"{model.output:10s}",
            "SPair-71k",
            cfg.split,
            f"{cfg.eval_class:>13s}",
            num_instances_label,
            probe_info,
        ]
    )
    entry = build_result_entry(
        "spair",
        model,
        output_dir,
        cfg,
        {f"Recall@{thresh}_view_diff={vp_diff}": avg_recall[i].item() for i, vp_diff in enumerate(["0", "1", "2", "all"])},
        dataset="SPair-71k",
        split=str(cfg.split),
        eval_class=str(cfg.eval_class),
        num_instances=num_instances_value,
        probe=probe_info,
    )
    append_jsonl(
        resolve_results_path(cfg, "correspondence_spair_linear_probe.jsonl"),
        entry,
    )
