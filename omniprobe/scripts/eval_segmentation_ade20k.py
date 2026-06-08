"""
Evaluate a saved ADE20K linear segmentation checkpoint.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from omniprobe.datasets.ade20k import ADE20KDataConfig, build_ade20k_dataloaders
from omniprobe.datasets.coco import create_pascal_label_colormap
from omniprobe.runtime import (
    append_jsonl,
    artifact_dir,
    build_result_entry,
    resolve_output_dir,
    resolve_results_path,
)
from omniprobe.utils.eval_helpers import resolve_mean_std
from omniprobe.utils.metrics import confusion_matrix, compute_miou
from omniprobe.utils.progress import progress


def _dense_feature(feats, expected_dim: int):
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    if feats.dim() != 4:
        raise ValueError(f"Expected dense feature map, got {tuple(feats.shape)}")
    if feats.shape[1] != expected_dim and feats.shape[-1] == expected_dim:
        feats = feats.permute(0, 3, 1, 2).contiguous()
    return feats


def _denormalize(image: torch.Tensor, mean, std):
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)
    return image * std + mean


def _colorize(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, colormap.shape[0] - 1)
    return colormap[mask]


def save_visualizations(
    model,
    head,
    val_loader,
    dataset_cfg,
    device,
    feat_dim,
    save_dir: Path,
    num_samples: int = 6,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    colormap = create_pascal_label_colormap()
    saved = 0
    mean, std = resolve_mean_std(dataset_cfg.image_mean)
    head.eval()
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            masks = batch["mask"]
            batch_size = images.size(0)

            for b in range(batch_size):
                if saved >= num_samples:
                    return
                img_tensor = images[b : b + 1].to(device)
                mask_tensor = masks[b : b + 1].to(device)

                feats = _dense_feature(model(img_tensor), expected_dim=feat_dim)
                feats = F.interpolate(
                    feats,
                    size=(dataset_cfg.val_mask_size, dataset_cfg.val_mask_size),
                    mode="bilinear",
                    align_corners=False,
                )
                logits = head(feats)
                preds_lowres = logits.argmax(dim=1)
                preds = F.interpolate(
                    preds_lowres.unsqueeze(1).float(),
                    size=img_tensor.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()

                img = _denormalize(img_tensor[0].cpu(), mean, std)
                img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
                gt = mask_tensor[0].cpu().numpy()
                pred = preds[0].cpu().numpy()
                gt_color = _colorize(gt, colormap)
                pred_color = _colorize(pred, colormap)
                panel = np.concatenate(
                    [
                        (img * 255).astype(np.uint8),
                        gt_color.astype(np.uint8),
                        pred_color.astype(np.uint8),
                    ],
                    axis=1,
                )
                Image.fromarray(panel).save(save_dir / f"eval_sample_{saved:02d}.jpg")
                saved += 1


class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(feats)


def run_task(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.system.random_seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = ADE20KDataConfig(**OmegaConf.to_container(cfg.dataset, resolve=True))
    _, val_loader = build_ade20k_dataloaders(dataset_cfg)

    backbone_kwargs = dict(output="dense")
    model = instantiate(cfg.backbone, **backbone_kwargs).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    feat_dim = model.feat_dim
    if isinstance(feat_dim, (list, tuple)):
        feat_dim = feat_dim[-1]

    head = LinearSegmentationHead(feat_dim, dataset_cfg.num_classes).to(device)
    if cfg.checkpoint_path is None:
        raise ValueError(
            "segmentation_ade20k_eval requires task.checkpoint_path pointing to a trained checkpoint."
        )
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    head.load_state_dict(state["head_state"])

    conf_matrix = torch.zeros(
        (dataset_cfg.num_classes, dataset_cfg.num_classes),
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for batch in progress(val_loader, desc="ADE20K evaluation"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            feats = _dense_feature(model(images), expected_dim=feat_dim)
            feats = F.interpolate(
                feats,
                size=(dataset_cfg.val_mask_size, dataset_cfg.val_mask_size),
                mode="bilinear",
                align_corners=False,
            )
            logits = head(feats)
            preds = logits.argmax(dim=1)
            if dataset_cfg.val_mask_size != masks.shape[-1]:
                masks_eval = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=(dataset_cfg.val_mask_size, dataset_cfg.val_mask_size),
                    mode="nearest",
                ).squeeze(1).long()
            else:
                masks_eval = masks
            conf_matrix += confusion_matrix(
                preds, masks_eval, dataset_cfg.num_classes, dataset_cfg.ignore_index
            )

    miou = compute_miou(conf_matrix)
    logger.info(f"Evaluation complete | mIoU: {miou:.4f}")

    if cfg.visualize:
        viz_dir = (
            Path(cfg.visualization_dir)
            if cfg.visualization_dir is not None
            else artifact_dir(cfg, "visualizations")
        )
        save_visualizations(
            model=model,
            head=head,
            val_loader=val_loader,
            dataset_cfg=dataset_cfg,
            device=device,
            feat_dim=feat_dim,
            save_dir=viz_dir,
            num_samples=cfg.num_visualizations,
        )
        logger.info(f"Saved {cfg.num_visualizations} qualitative samples to {viz_dir}")

    results_path = resolve_results_path(cfg, "segmentation_ade20k_eval.jsonl")
    entry = build_result_entry(
        "ade20k",
        model,
        resolve_output_dir(cfg),
        cfg,
        {"mIoU": miou},
        dataset="ADE20K",
        head="linear",
    )
    append_jsonl(results_path, entry)
