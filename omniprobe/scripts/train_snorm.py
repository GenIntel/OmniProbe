import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from omniprobe.datasets.builder import build_loader
from omniprobe.runtime import (
    append_jsonl,
    artifact_dir,
    build_result_entry,
    resolve_output_dir,
    resolve_results_path,
)
from omniprobe.utils.losses import angular_loss
from omniprobe.utils.metrics import evaluate_surface_norm
from omniprobe.utils.optim import cosine_decay_linear_warmup
from omniprobe.utils.progress import progress
from omniprobe.utils.training import ddp_setup, ddp_cleanup, set_seed, unwrap_model


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    device,
    rank=0,
    world_size=1,
    valid_loader=None,
    max_steps_per_epoch=None,
):
    for ep in range(n_epochs):

        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = progress(train_loader, desc=f"Epoch {ep}") if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
                break
            images = batch["image"].to(device)
            mask = batch["depth"].to(device) > 0
            target = batch["snorm"].to(device)

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if isinstance(feats, (tuple, list)):
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()

            else:
                feats = model(images)
            pred = probe(feats)
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")

            uncertainty = pred.shape[1] > 3
            loss = angular_loss(pred, target, mask, uncertainty_aware=uncertainty)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(f"{ep} | loss: {_loss:.4f} probe_lr: {pr_lr:.2e}")

        total_steps = (
            min(len(train_loader), max_steps_per_epoch)
            if max_steps_per_epoch is not None
            else len(train_loader)
        )
        train_loss /= total_steps

        if rank == 0 and valid_loader is not None:
            valid_loss, valid_metrics = validate(
                model,
                probe,
                valid_loader,
                device,
            )
            logger.info(f"Final valid loss       | {valid_loss:.4f}")
            for metric in valid_metrics:
                logger.info(f"Final valid {metric:10s} | {valid_metrics[metric]:.4f}")


def validate(model, probe, loader, device, verbose=True, aggregate=True, max_batches=None):
    total_loss = 0.0
    metrics = None
    with torch.inference_mode():
        pbar = progress(loader, desc="Evaluation") if verbose else loader
        for idx, batch in enumerate(pbar):
            if max_batches is not None and idx >= max_batches:
                break
            images = batch["image"].to(device)
            mask = batch["depth"].to(device) > 0
            target = batch["snorm"].to(device)

            feats = model(images)
            pred = probe(feats)
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")

            uncertainty = pred.shape[1] > 3
            loss = angular_loss(pred, target, mask, uncertainty_aware=uncertainty)

            total_loss += loss.item()
            batch_metrics = evaluate_surface_norm(pred.detach(), target, mask)
            if metrics is None:
                metrics = {key: [batch_metrics[key]] for key in batch_metrics}
            else:
                for key in batch_metrics:
                    metrics[key].append(batch_metrics[key])

    # aggregate
    total_steps = min(len(loader), max_batches) if max_batches is not None else len(loader)
    total_loss = total_loss / total_steps
    for key in metrics:
        metric_key = torch.cat(metrics[key], dim=0)
        metrics[key] = metric_key.mean() if aggregate else metric_key

    return total_loss, metrics


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    seed = int(cfg.system.random_seed)

    # ===== GET DATA LOADERS =====
    # validate and test on single gpu
    trainval_loader = build_loader(
        cfg.dataset,
        "trainval",
        cfg.batch_size,
        world_size,
        seed=seed,
    )
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    trainval_loader.dataset.__getitem__(0)

    # ===== Get models =====
    model = instantiate(cfg.backbone)
    set_seed(seed)
    probe = instantiate(cfg.probe, feat_dim=model.feat_dim)

    # === job info
    train_dset = trainval_loader.dataset.name
    test_dset = test_loader.dataset.name
    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{cfg.optimizer.probe_lr:4.2e}",
        f"{cfg.optimizer.model_lr:4.2e}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]
    output_dir = resolve_output_dir(cfg)

    # ===== SETUP LOGGING =====
    if rank == 0:
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(device)
    probe = probe.to(device)

    # SAM / ViT-MAE need a fixed input size under DDP finetuning
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = trainval_loader.dataset.__getitem__(0)["image"].shape[-2:]
        model.resize_pos_embed(image_size=(h, w))

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    max_steps_cfg = getattr(cfg.system, "max_steps_per_epoch", 0)
    if max_steps_cfg and max_steps_cfg > 0:
        steps_per_epoch = min(len(trainval_loader), int(max_steps_cfg))
        max_steps_per_epoch = steps_per_epoch
    else:
        steps_per_epoch = len(trainval_loader)
        max_steps_per_epoch = None

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(
        epoch,
        cfg.optimizer.n_epochs * steps_per_epoch,
        cfg.optimizer.warmup_epochs * steps_per_epoch,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)

    max_eval_batches = getattr(cfg.system, "max_eval_batches", 0)
    if max_eval_batches and max_eval_batches > 0:
        max_eval_batches = int(max_eval_batches)
    else:
        max_eval_batches = None

    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        device=device,
        rank=rank,
        world_size=world_size,
        # valid_loader=test_loader,
        max_steps_per_epoch=max_steps_per_epoch,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        test_loss, test_metrics = validate(
            model,
            probe,
            test_loader,
            device,
            max_batches=max_eval_batches,
        )
        logger.info(f"Final test loss       | {test_loss:.4f}")
        for metric in test_metrics:
            logger.info(f"Final test {metric:10s} | {test_metrics[metric]:.4f}")

        serializable_metrics = {
            metric: float(test_metrics[metric]) for metric in test_metrics
        }

        # result summary
        model_info = ", ".join(model_info)
        probe_info = ", ".join(probe_info)
        train_info = ", ".join(train_info)
        results = ", ".join([f"{serializable_metrics[_m]:.4f}" for _m in serializable_metrics])

        entry = build_result_entry(
            "snorm",
            model,
            output_dir,
            cfg,
            serializable_metrics,
            probe=probe_info,
            training=train_info,
            test_dataset=test_dset,
        )
        append_jsonl(
            resolve_results_path(cfg, f"snorm_{test_dset}.jsonl"),
            entry,
        )

        # save final model
        ckpt_path = artifact_dir(cfg, "checkpoints") / "ckpt.pth"
        checkpoint = {
            "cfg": cfg,
            "model": unwrap_model(model).state_dict(),
            "probe": unwrap_model(probe).state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint at {ckpt_path}")

    ddp_cleanup(world_size)


def run_task(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)
