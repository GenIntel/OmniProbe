from datetime import datetime

import torch
from loguru import logger

from omniprobe.models.contracts import (
    get_backbone_contract,
    instantiate_backbone_for_output,
    uses_multilayer_global_features,
)
from omniprobe.runtime import (
    append_jsonl,
    build_result_entry,
    config_to_string,
    extract_backbone_features,
    log_runtime_header,
    resolve_results_path,
)
from omniprobe.tasks.imagenet_common import build_imagenet_loaders


TASK_NAME = "imagenet_knn"
SUPPORTED_MODES = ("default",)
REQUIRED_SAMPLE_SCHEMA = "Tuple[image, class_index]"
REQUIRED_BACKBONE_OUTPUTS = ("cls", "gap", "map")


def _extract_features(model, loader, device, split_name: str = "split"):
    features = []
    targets = []
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    dataset = loader.dataset if hasattr(loader, "dataset") else None
    total_samples = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else None
    seen_samples = 0
    message = f"[imagenet_knn] Starting feature extraction for {split_name}"
    if total_samples is not None and total_batches is not None:
        message += f": {total_samples} images across {total_batches} batches"
    logger.info(message)
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            feats = extract_backbone_features(model, images, pool="global")
            features.append(feats.cpu())
            targets.append(labels)
            seen_samples += labels.shape[0]
            if batch_idx == 1:
                logger.info(
                    f"[imagenet_knn] {split_name}: first batch feature shape={tuple(feats.shape)}"
                )
            is_last_batch = total_batches is not None and batch_idx == total_batches
            if batch_idx % 25 != 0 and not is_last_batch:
                continue
            progress = f"[imagenet_knn] {split_name}: processed batch {batch_idx}"
            if total_batches is not None:
                progress = f"[imagenet_knn] {split_name}: processed {batch_idx}/{total_batches} batches"
            if total_samples is not None:
                progress += f" ({seen_samples}/{total_samples} images)"
            logger.info(progress)
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _accuracy(probs, targets, topk=(1, 5)):
    maxk = min(max(topk), probs.shape[1])
    _, pred = probs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    results = []
    for k in topk:
        k = min(k, probs.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.item() * (100.0 / targets.size(0)))
    return results


def _knn_predict(train_feats, train_labels, val_feats, val_targets, k_list, temperature, num_classes):
    import faiss

    train_feats = torch.nn.functional.normalize(train_feats, dim=1)
    val_feats = torch.nn.functional.normalize(val_feats, dim=1)

    train_feats_np = train_feats.to(dtype=torch.float32).numpy()
    val_feats_np = val_feats.to(dtype=torch.float32).numpy()
    train_labels_np = train_labels.to(dtype=torch.int64).numpy()
    max_k = max(k_list)

    logger.info(
        f"[imagenet_knn] Building FAISS index for {train_feats_np.shape[0]} train features"
    )
    index = faiss.IndexFlatIP(train_feats_np.shape[1])
    index.add(train_feats_np)
    logger.info(
        f"[imagenet_knn] Searching {val_feats_np.shape[0]} val features with max_k={max_k}"
    )
    top_sims_np, top_indices_np = index.search(val_feats_np, max_k)

    top_sims = torch.from_numpy(top_sims_np)
    neighbor_labels = torch.from_numpy(train_labels_np[top_indices_np])

    results = {}
    for k in k_list:
        partial_sims = top_sims[:, :k]
        partial_labels = neighbor_labels[:, :k]
        weights = torch.softmax(partial_sims / temperature, dim=1)
        probs = torch.zeros(val_feats.size(0), num_classes)
        for index in range(k):
            probs.scatter_add_(
                1,
                partial_labels[:, index : index + 1],
                weights[:, index : index + 1],
            )
        top1, top5 = _accuracy(probs, val_targets, topk=(1, 5))
        results[int(k)] = {"top1": top1, "top5": top5}
    return results


def run(cfg, context):
    log_runtime_header(TASK_NAME, "default", context)
    contract = get_backbone_contract(cfg.backbone)
    output_name = contract.resolve_global_output()
    if "output" in cfg.task and cfg.task.output is not None:
        output_name = str(cfg.task.output)
        contract.require_output(output_name, TASK_NAME, "default")
    model, contract = instantiate_backbone_for_output(
        cfg.backbone,
        output_name=output_name,
        return_multilayer=uses_multilayer_global_features(cfg.backbone),
        device=context.device,
    )
    train_loader, val_loader = build_imagenet_loaders(cfg.task, contract)
    num_classes = len(train_loader.dataset.classes)

    train_feats, train_labels = _extract_features(model, train_loader, context.device, "train")
    val_feats, val_labels = _extract_features(model, val_loader, context.device, "val")
    results = _knn_predict(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        list(cfg.task.knn_k),
        float(cfg.task.temperature),
        num_classes,
    )
    log_path = resolve_results_path(cfg, "imagenet_knn.jsonl")
    entry = build_result_entry(
        TASK_NAME,
        "default",
        model,
        context.output_dir,
        cfg,
        results,
        dataset="ImageNet",
        split=f"{cfg.task.train_split}->{cfg.task.val_split}",
    )
    append_jsonl(log_path, entry)
    logger.info(f"KNN results: {results}")
    return entry
