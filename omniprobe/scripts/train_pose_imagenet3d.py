import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from omniprobe.datasets.builder import build_loader
from omniprobe.models.contracts import get_backbone_contract, instantiate_backbone_for_output
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.utils.pose import bin_to_continuous, batch_pose_error


def train_one_epoch(
    backbone,
    probe,
    loader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    log_freq,
):
    probe.train()
    backbone.eval()
    running_loss = 0.0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        az_idx = batch["azimuth_idx"].to(device, non_blocking=True)
        el_idx = batch["elevation_idx"].to(device, non_blocking=True)
        th_idx = batch["theta_idx"].to(device, non_blocking=True)

        with torch.no_grad():
            feats = backbone(images)
        logits = probe(feats)

        loss = (
            loss_fn(logits[0], az_idx)
            + loss_fn(logits[1], el_idx)
            + loss_fn(logits[2], th_idx)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        if (step + 1) % log_freq == 0:
            logger.info(
                f"[train] step {step+1}/{len(loader)} "
                f"loss={running_loss / (step + 1):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(backbone, probe, loader, device, multi_bin_cfg):
    backbone.eval()
    probe.eval()

    all_errors = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        az = batch["azimuth"].cpu().numpy()
        el = batch["elevation"].cpu().numpy()
        th = batch["theta"].cpu().numpy()

        feats = backbone(images)
        logits = probe(feats)

        az_pred_bins = torch.argmax(logits[0], dim=1).cpu()
        el_pred_bins = torch.argmax(logits[1], dim=1).cpu()
        th_pred_bins = torch.argmax(logits[2], dim=1).cpu()

        az_pred = bin_to_continuous(
            az_pred_bins,
            num_bins=multi_bin_cfg.num_bins,
            min_value=multi_bin_cfg.min_value,
            max_value=multi_bin_cfg.max_value,
        )
        el_pred = bin_to_continuous(
            el_pred_bins,
            num_bins=multi_bin_cfg.num_bins,
            min_value=multi_bin_cfg.min_value,
            max_value=multi_bin_cfg.max_value,
        )
        th_pred = bin_to_continuous(
            th_pred_bins,
            num_bins=multi_bin_cfg.num_bins,
            min_value=multi_bin_cfg.min_value,
            max_value=multi_bin_cfg.max_value,
        )

        errors = batch_pose_error(
            (az, el, th),
            (az_pred, el_pred, th_pred),
        )
        all_errors.append(errors)

    if not all_errors:
        raise RuntimeError("Validation loader returned no batches.")

    errors = np.concatenate([np.real(err) for err in all_errors])
    errors_tensor = torch.from_numpy(errors)
    pi_6 = (errors_tensor < torch.pi / 6).float().mean().item()
    pi_18 = (errors_tensor < torch.pi / 18).float().mean().item()
    med_err = errors_tensor.median().item() * 180.0 / torch.pi
    mean_err = errors_tensor.mean().item() * 180.0 / torch.pi

    return {
        "pi_6_acc": pi_6,
        "pi_18_acc": pi_18,
        "median_error_deg": med_err,
        "mean_error_deg": mean_err,
    }


def prepare_backbone(cfg, output_name: str | None = None):
    contract = get_backbone_contract(cfg)
    if output_name is None:
        output_name = contract.resolve_global_output()
    backbone, _ = instantiate_backbone_for_output(
        cfg,
        output_name=output_name,
        return_multilayer=True,
        device=torch.device("cpu"),
    )
    logger.info(f"Using {output_name} backbone output.")
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def run_task(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.system.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(HydraConfig.get().run.dir)
    logger.info(f"Writing logs and checkpoints to {output_dir}")

    train_loader = build_loader(
        cfg.dataset,
        "train",
        cfg.training.batch_size,
        1,
        num_bins=cfg.multi_bin.num_bins,
        min_angle=cfg.multi_bin.min_value,
        max_angle=cfg.multi_bin.max_value,
    )
    val_loader = build_loader(
        cfg.dataset,
        "val",
        cfg.evaluation.batch_size,
        1,
        num_bins=cfg.multi_bin.num_bins,
        min_angle=cfg.multi_bin.min_value,
        max_angle=cfg.multi_bin.max_value,
    )

    is_ep = "pose_ep" in str(cfg.probe.get("_target_", ""))
    backbone = prepare_backbone(cfg.backbone, output_name="dense" if is_ep else None)
    backbone = backbone.to(device)

    feat_dims = backbone.feat_dim
    if isinstance(feat_dims, int):
        feat_dims = [feat_dims]

    probe = instantiate(
        cfg.probe,
        feat_dims=feat_dims,
        num_bins=cfg.multi_bin.num_bins,
    ).to(device)

    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.training.epochs * len(train_loader)),
        eta_min=cfg.training.min_lr,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.training.epochs):
        logger.info(f"Starting epoch {epoch+1}/{cfg.training.epochs}")
        avg_loss = train_one_epoch(
            backbone,
            probe,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_fn,
            cfg.training.log_freq,
        )
        logger.info(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f}")

    metrics = evaluate(backbone, probe, val_loader, device, cfg.multi_bin)
    logger.info(
        "Validation metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )

    ckpt_path = output_dir / "pose_probe_final.pth"
    torch.save(
        {
            "probe": probe.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        ckpt_path,
    )

    logger.info(f"Saving to {output_dir}")
    with open(output_dir / "metrics_v2.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    entry = build_result_entry(
        "imagenet3d_pose",
        "default",
        backbone,
        output_dir,
        cfg,
        metrics,
        dataset="ImageNet3D",
    )
    append_jsonl(resolve_results_path(cfg, "pose_imagenet3d.jsonl"), entry)
