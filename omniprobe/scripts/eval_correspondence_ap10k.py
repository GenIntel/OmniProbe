from datetime import datetime
import json
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn_F
from einops import einsum
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from omniprobe.datasets.ap10k import AP10KDataset, get_ap10k_categories
from omniprobe.runtime import append_jsonl, build_result_entry, resolve_results_path
from omniprobe.utils.correspondence import argmax_2d
from omniprobe.utils.paths import cfg_or_env_path

from hydra.core.hydra_config import HydraConfig
import os

# ========== Helper ==========

def to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def visualize_matching(img_a, img_b, kps_a, kps_b, pred_kps, heatmaps, keypoint_idx=0, save_path=None):
    img_a_np = to_numpy(img_a)
    img_b_np = to_numpy(img_b)

    H, W = img_a_np.shape[:2]
    pt_a = (kps_a[keypoint_idx, 0] * W, kps_a[keypoint_idx, 1] * H)
    pt_b_gt = (kps_b[keypoint_idx, 0] * W, kps_b[keypoint_idx, 1] * H)
    pt_b_pred = (pred_kps[keypoint_idx, 0] * W, pred_kps[keypoint_idx, 1] * H)

    img_a_draw = img_a_np.copy()
    img_b_draw = img_b_np.copy()
    cv2.circle(img_a_draw, (int(pt_a[0]), int(pt_a[1])), 5, (1, 0, 0), -1)
    cv2.circle(img_b_draw, (int(pt_b_gt[0]), int(pt_b_gt[1])), 5, (0, 1, 0), -1)
    cv2.circle(img_b_draw, (int(pt_b_pred[0]), int(pt_b_pred[1])), 5, (1, 0, 0), -1)

    heatmap = heatmaps[keypoint_idx].cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    heatmap_overlay = (0.4 * heatmap_colored + 0.6 * img_b_np)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img_a_draw)
    axes[0].set_title("Image A with Keypoint")
    axes[1].imshow(img_b_draw)
    axes[1].set_title("Image B with GT (Green) and Pred (Red)")
    axes[2].imshow(heatmap_overlay)
    axes[2].set_title("Matching Heatmap on Image B")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def _serialize_pred_output(pred_output):
    """Convert tensors in pred_output to Python lists for JSON dumps."""
    out = {}
    for key, value in pred_output.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().tolist()
        else:
            out[key] = value
    return out


def compute_predictions(model, instance, mask_feats=False, return_heatmaps=False):
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, class_name = instance
    device = next(model.parameters()).device
    mask_i = torch.tensor(np.array(mask_i, dtype=float))
    mask_j = torch.tensor(np.array(mask_j, dtype=float))

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


def compute_errors(model, instance, mask_feats=False, return_heatmaps=False):
    pred_output = compute_predictions(model, instance, mask_feats, return_heatmaps)
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


def evaluate_dataset(
    model,
    dataset,
    thresh,
    verbose=False,
    log_fh=None,
    class_name=None,
    subset_name=None,
    record_list=None,
):
    iterator = tqdm(range(len(dataset)), ncols=60) if verbose else range(len(dataset))
    errors_all = []
    src_all = []
    tgt_all = []

    for idx in iterator:
        error_same, _, index_same, index_nn, pred_output = compute_errors(
            model, dataset.__getitem__(idx)
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
                meta["subset"] = subset_name
            if hasattr(dataset, "instances"):
                pair_info = dataset.instances[idx]
                if isinstance(pair_info, dict):
                    if "pair_file" in pair_info:
                        meta.setdefault("pair_filename", pair_info["pair_file"])
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


def run_task(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir
    print(f'Output dir: {output_dir}')
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    data_root = cfg_or_env_path(cfg, "data_root", "AP10K_ROOT", "AP-10K dataset root")
    thresh = cfg.get("thresh", 0.10)
    eval_subset = cfg.get("eval_subset", "intra-species")

    # ===== Get model =====
    backbone_kwargs = dict(output="dense")
    if cfg.multilayer:
        backbone_kwargs["return_multilayer"] = True
    
    device = torch.device(cfg.device) if "device" in cfg else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = instantiate(cfg.backbone, **backbone_kwargs)
    model = model.to(device)

    # ===== GET CATEGORIES =====
    if cfg.eval_class == "all":
        classes, modified_split = get_ap10k_categories(data_root, eval_subset, cfg.split)
    else:
        classes = [cfg.eval_class]
        _, modified_split = get_ap10k_categories(data_root, eval_subset, cfg.split)

    logger.info(f"Evaluating on {len(classes)} classes with eval_subset={eval_subset}")
    logger.info(f"Modified split: {modified_split}")

    pred_log_path = os.path.join(output_dir, "pred_outputs_ap10k_correspondence.json")
    pred_pkl_path = os.path.join(output_dir, "pred_outputs_ap10k_correspondence.pkl")
    logger.info(f"Logging per-pair predictions to {pred_log_path}")

    class_acc = {}
    pred_records = []
    
    with open(pred_log_path, "w") as pred_log_fh:
        for class_name in classes:
            dataset = AP10KDataset(
                data_root,
                cfg.split,
                use_bbox=cfg.use_bbox,
                image_size=cfg.image_size,
                image_mean=cfg.image_mean,
                class_name=class_name,
                num_instances=cfg.num_instances,
                eval_subset=eval_subset,
            )
            
            if len(dataset) > 0:
                recall, confusion = evaluate_dataset(
                    model,
                    dataset,
                    thresh,
                    verbose=True,
                    log_fh=pred_log_fh,
                    class_name=class_name,
                    subset_name=eval_subset,
                    record_list=pred_records,
                )
                logger.info(
                    f"Recall@{thresh} {class_name:>20s} |  {recall:6.2f} ({len(dataset)} pairs)"
                )
            else:
                logger.info(f"Recall@{thresh} {class_name:>20s} |  N/A (0 pairs)")
                recall, confusion = -1, None
            
            class_acc[class_name] = (recall, confusion)

    with open(pred_pkl_path, "wb") as pred_pkl_fh:
        pickle.dump(pred_records, pred_pkl_fh)
    logger.info(f"Wrote pickle predictions to {pred_pkl_path}")

    # Compute average recall
    all_recall = [class_acc[cls][0] for cls in class_acc if class_acc[cls][0] >= 0]
    if all_recall:
        avg_recall = sum(all_recall) / len(all_recall)
    else:
        avg_recall = float('nan')

    logger.info(f"Average Recall@{thresh}: {avg_recall:6.2f}")

    entry = build_result_entry(
        "ap10k",
        "default",
        model,
        output_dir,
        cfg,
        {"recall@0.10": avg_recall},
        dataset="AP-10K",
        split=str(cfg.split),
        eval_subset=eval_subset,
        eval_class=str(cfg.eval_class),
        num_instances=int(cfg.num_instances),
    )
    append_jsonl(resolve_results_path(cfg, "correspondence_ap10k.jsonl"), entry)
