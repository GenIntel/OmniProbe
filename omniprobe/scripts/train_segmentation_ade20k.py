"""
Train a linear segmentation probe on ADE20K using frozen backbones.

This mirrors the NeCo linear finetuning recipe (RandomResizedCrop, SGD + StepLR,
25 epochs, lr 0.01, drop at epoch 20) while integrating with the OmniProbe
backbone/config infrastructure.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from omniprobe.datasets.ade20k import ADE20KDataConfig, build_ade20k_dataloaders
from omniprobe.datasets.coco import create_pascal_label_colormap
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.utils.eval_helpers import resolve_mean_std
from omniprobe.utils.metrics import confusion_matrix, compute_miou


def _dense_feature(feats, expected_dim: int):
    """
    Normalize the backbone outputs to a single (B, C, H, W) tensor.
    """
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    if feats.dim() != 4:
        raise ValueError(f"Expected dense feature map, got shape {tuple(feats.shape)}")

    # Handle channel-last layouts that some backbones may emit.
    if feats.shape[1] != expected_dim and feats.shape[-1] == expected_dim:
        feats = feats.permute(0, 3, 1, 2).contiguous()
    return feats


class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, use_bn: bool = False):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.bn(feats))


def _denormalize(image: torch.Tensor, mean, std) -> torch.Tensor:
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
    num_samples: int = 4,
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
                    size=img_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                logits = head(feats)
                preds = logits.argmax(dim=1)

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
                out_path = save_dir / f"sample_{saved:02d}.png"
                Image.fromarray(panel).save(out_path)
                saved += 1


def train(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.system.random_seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = ADE20KDataConfig(**OmegaConf.to_container(cfg.dataset, resolve=True))
    train_loader, val_loader = build_ade20k_dataloaders(dataset_cfg)

    backbone_kwargs = dict(output="dense")
    model = instantiate(cfg.backbone, **backbone_kwargs).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    feat_dim = model.feat_dim
    if isinstance(feat_dim, (list, tuple)):
        feat_dim = feat_dim[-1]

    use_bn = getattr(cfg, 'use_bn_head', False)
    head = LinearSegmentationHead(feat_dim, dataset_cfg.num_classes, use_bn=use_bn).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_cfg.ignore_index)
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.optimizer.drop_at,
        gamma=cfg.optimizer.decay_rate,
    )

    # Setup checkpointing
    run_dir = Path(os.getcwd())
    ckpt_dir = Path(cfg.checkpoint.dir) if cfg.checkpoint.dir else run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.ckpt"
    best_ckpt = ckpt_dir / "best.ckpt"

    results_path = resolve_results_path(cfg, "segmentation_ade20k_train.jsonl")

    start_epoch = 0
    best_miou = 0.0
    resume_path = cfg.checkpoint.resume_path
    if cfg.checkpoint.resume:
        if resume_path is None and last_ckpt.exists():
            resume_path = last_ckpt
        if resume_path is not None and Path(resume_path).exists():
            state = torch.load(resume_path, map_location=device)
            head.load_state_dict(state["head_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            best_miou = state.get("best_miou", 0.0)
            start_epoch = state.get("epoch", 0) + 1
            logger.info(
                f"Resumed from {resume_path} | start_epoch={start_epoch}, best_mIoU={best_miou:.4f}"
            )
        elif resume_path is not None:
            logger.warning(f"Resume path {resume_path} not found. Starting from scratch.")
    else:
        logger.info(f"Starting from scratch.")

    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        head.train()
        epoch_loss = 0.0
        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.optimizer.max_epochs}", ncols=100)
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.no_grad():
                if hasattr(model, "dense_with_prompts"):
                    feats = _dense_feature(
                        model.dense_with_prompts(images, prompts=[""] * images.size(0)),
                        expected_dim=feat_dim,
                    )
                else:
                    feats = _dense_feature(model(images), expected_dim=feat_dim)
            feats = F.interpolate(
                feats,
                size=(dataset_cfg.train_mask_size, dataset_cfg.train_mask_size),
                mode="bilinear",
                align_corners=False,
            )
            logits = head(feats)

            if dataset_cfg.train_mask_size != masks.shape[-1]:
                masks_resized = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=(dataset_cfg.train_mask_size, dataset_cfg.train_mask_size),
                    mode="nearest",
                ).squeeze(1).long()
            else:
                masks_resized = masks

            loss = criterion(logits, masks_resized)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loader.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        logger.info(f"[Epoch {epoch+1}] Train loss: {avg_loss:.4f}")

        head.eval()
        conf_matrix = torch.zeros(
            (dataset_cfg.num_classes, dataset_cfg.num_classes),
            device=device,
            dtype=torch.float32,
        )
        batches_evaluated = 0
        with torch.no_grad():
            for batch in val_loader:
                if (
                    dataset_cfg.val_iters is not None
                    and batches_evaluated >= dataset_cfg.val_iters
                ):
                    break
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)

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
                batches_evaluated += 1

        miou = compute_miou(conf_matrix)
        logger.info(f"[Epoch {epoch+1}] Validation mIoU: {miou:.4f}")

        improved = miou > best_miou
        if improved:
            best_miou = miou

        checkpoint_payload = {
            "epoch": epoch,
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": best_miou,
        }
        torch.save(checkpoint_payload, last_ckpt)
        if improved:
            torch.save(checkpoint_payload, best_ckpt)
            viz_dir = ckpt_dir / "visualizations"
            save_visualizations(
                model=model,
                head=head,
                val_loader=val_loader,
                dataset_cfg=dataset_cfg,
                device=device,
                feat_dim=feat_dim,
                save_dir=viz_dir,
                num_samples=4,
            )

    entry = build_result_entry(
        "ade20k",
        "train",
        model,
        run_dir,
        cfg,
        {"best_mIoU": best_miou},
        dataset="ADE20K",
        head="linear",
    )
    append_jsonl(results_path, entry)


def run_task(cfg: DictConfig) -> None:
    train(cfg)
