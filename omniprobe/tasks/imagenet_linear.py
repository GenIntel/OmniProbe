from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
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


TASK_NAME = "imagenet_linear"
SUPPORTED_MODES = ("default",)
REQUIRED_SAMPLE_SCHEMA = "Tuple[image, class_index]"
REQUIRED_BACKBONE_OUTPUTS = ("cls", "gap", "map")
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def _accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / target.size(0)).item())
    return results


def _create_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_feature(model, images):
    return extract_backbone_features(model, images, pool="global")


def _evaluate(model, head, loader, device):
    head.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            feats = _compute_feature(model, images)
            logits = head(feats)
            acc1, acc5 = _accuracy(logits, target)
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
    return top1.avg, top5.avg


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
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    images, _ = next(iter(train_loader))
    images = images.to(context.device)
    with torch.no_grad():
        feat_dim = _compute_feature(model, images).shape[1]

    head = nn.Linear(feat_dim, num_classes).to(context.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.task.label_smoothing))
    optimizer = optim.SGD(
        head.parameters(),
        lr=float(cfg.task.lr),
        momentum=float(cfg.task.momentum),
        weight_decay=float(cfg.task.weight_decay),
    )
    total_steps = int(cfg.task.epochs) * len(train_loader)
    warmup_steps = int(float(cfg.task.warmup_epochs) * len(train_loader))
    scheduler = _create_scheduler(optimizer, total_steps, warmup_steps)

    best_top1 = 0.0
    for epoch in range(int(cfg.task.epochs)):
        head.train()
        losses = AverageMeter()
        for images, target in train_loader:
            images = images.to(context.device, non_blocking=True)
            target = target.to(context.device, non_blocking=True)
            with torch.no_grad():
                feats = _compute_feature(model, images)
            logits = head(feats)
            loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.item(), images.size(0))
        acc1, acc5 = _evaluate(model, head, val_loader, context.device)
        logger.info(
            f"Epoch {epoch + 1}/{cfg.task.epochs}: "
            f"loss={losses.avg:.4f}, top1={acc1:.2f}, top5={acc5:.2f}"
        )
        if acc1 > best_top1:
            best_top1 = acc1

    entry = build_result_entry(
        TASK_NAME,
        "default",
        model,
        context.output_dir,
        cfg,
        {"top1": best_top1},
        dataset="ImageNet",
        split=f"{cfg.task.train_split}->{cfg.task.val_split}",
        epochs=int(cfg.task.epochs),
    )
    append_jsonl(resolve_results_path(cfg, "imagenet_linear.jsonl"), entry)
    return entry
