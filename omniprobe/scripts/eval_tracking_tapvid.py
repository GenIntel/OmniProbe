"""Zero-shot TAP-Vid tracking evaluation for OmniProbe backbones."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from omniprobe.datasets.builder import build_loader
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.utils.eval_helpers import resolve_mean_std
from omniprobe.utils.tapvid import (
    TapVidEvaluator,
    compute_cost_volume,
    compute_tapvid_metrics,
)


def _chunk_backbone(
    model: torch.nn.Module, frames: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for chunk in frames.split(chunk_size, dim=0):
            feats = model(chunk)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            outputs.append(feats)
    return torch.cat(outputs, dim=0)


def _sample_query_tokens(
    tokens: torch.Tensor,
    queries: torch.Tensor,
    processed_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Sample query descriptors from dense tokens by bilinear interpolation.
    """
    b, _, c, h, w = tokens.shape
    _, n, _ = queries.shape
    device = tokens.device

    query_embeds = torch.zeros(b, n, c, device=device)
    pw, ph = processed_hw[1], processed_hw[0]

    for batch_idx in range(b):
        q = queries[batch_idx]
        frame_index = q[:, 0].long()
        selected = tokens[batch_idx, frame_index]  # (N, C, H, W)
        feat_h, feat_w = selected.shape[-2:]

        x_feat = (q[:, 1] / pw) * (feat_w - 1)
        y_feat = (q[:, 2] / ph) * (feat_h - 1)

        denom_x = feat_w - 1 if feat_w > 1 else 1.0
        denom_y = feat_h - 1 if feat_h > 1 else 1.0

        x = (x_feat / denom_x) * 2 - 1
        y = (y_feat / denom_y) * 2 - 1

        grid = torch.stack([x, y], dim=-1).view(-1, 1, 1, 2)
        sampled = F.grid_sample(
            selected,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        query_embeds[batch_idx] = sampled.squeeze(-1).squeeze(-1)

    return query_embeds


def run_task(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.random_seed)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(HydraConfig.get().run.dir)
    logger.info(f"Writing TAP-Vid outputs to {output_dir}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.device) if "device" in cfg else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    mean, std = resolve_mean_std(cfg.image_mean)
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    loader = build_loader(
        cfg.dataset,
        split="test",
        batch_size=1,
        num_workers=cfg.dataloader_workers,
    )

    backbone_kwargs = dict(output="dense")
    if "multilayer" in cfg and cfg.multilayer:
        backbone_kwargs["return_multilayer"] = True
    model = instantiate(cfg.backbone, **backbone_kwargs)
    model = model.to(device)
    model.eval()

    evaluator = TapVidEvaluator(zero_shot=True)
    processed_hw = tuple(cfg.feature.input_size)
    chunk_size = cfg.feature.frame_chunk

    for sample in loader:
        video = sample["video"].to(device)  # (B, T, 3, H, W)
        traj = sample["trajectory"].to(device)
        visibility = sample["visibility"].to(device)
        queries = sample["query_points"].to(device)
        b, t, _, orig_h, orig_w = video.shape

        flat_video = video.view(b * t, 3, orig_h, orig_w)
        if processed_hw is not None:
            flat_video = F.interpolate(
                flat_video, size=processed_hw, mode="bilinear", align_corners=False
            )
        else:
            processed_hw = (orig_h, orig_w)
        flat_video = (flat_video - mean) / std

        dense_feats = _chunk_backbone(model, flat_video, chunk_size)
        _, c, feat_h, feat_w = dense_feats.shape
        tokens = dense_feats.view(b, t, c, feat_h, feat_w)

        scale_x = processed_hw[1] / orig_w
        scale_y = processed_hw[0] / orig_h
        scaled_queries = queries.clone()
        scaled_queries[:, :, 1] = scaled_queries[:, :, 1] * scale_y
        scaled_queries[:, :, 2] = scaled_queries[:, :, 2] * scale_x
        tap_queries = torch.stack(
            [scaled_queries[:, :, 0], scaled_queries[:, :, 2], scaled_queries[:, :, 1]],
            dim=-1,
        )
        query_tokens = _sample_query_tokens(tokens, tap_queries, processed_hw)

        cost_volumes = compute_cost_volume(tokens, query_tokens)
        flat = cost_volumes.view(b, t, queries.size(1), -1)
        max_idx = flat.argmax(-1)
        max_x = max_idx % feat_w
        max_y = max_idx // feat_w
        coords = torch.stack([max_x, max_y], dim=-1).float() + 0.5

        coords[..., 0] *= processed_hw[1] / feat_w
        coords[..., 1] *= processed_hw[0] / feat_h
        coords[..., 0] *= orig_w / processed_hw[1]
        coords[..., 1] *= orig_h / processed_hw[0]

        query_np = queries.cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occ = torch.logical_not(visibility).permute(0, 2, 1).cpu().numpy()
        pred_tracks = coords.permute(0, 2, 1, 3).cpu().numpy()
        pred_occ = np.zeros_like(gt_occ, dtype=bool)

        metrics = compute_tapvid_metrics(
            query_np,
            gt_occ,
            gt_tracks,
            pred_occ,
            pred_tracks,
            loader.dataset.query_mode,
        )
        evaluator.update(metrics)

    evaluator.report()

    summary = build_result_entry(
            "tapvid",
            model,
            output_dir,
            cfg,
            {
                "delta_avg": float(np.mean(evaluator.delta_avg)),
                "delta_1": float(np.mean(evaluator.delta_1)),
                "delta_2": float(np.mean(evaluator.delta_2)),
                "delta_4": float(np.mean(evaluator.delta_4)),
                "delta_8": float(np.mean(evaluator.delta_8)),
                "delta_16": float(np.mean(evaluator.delta_16)),
                "occ_acc": float(np.mean(evaluator.occlusion)),
                "jaccard": float(np.mean(evaluator.jaccard)),
            },
            dataset=loader.dataset.name,
    )
    log_path = resolve_results_path(cfg, "tracking_tapvid.jsonl")
    append_jsonl(log_path, summary)
