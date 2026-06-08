import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import interpolate
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
from omniprobe.utils.losses import DepthLoss
from omniprobe.utils.metrics import evaluate_depth, match_scale_and_shift
from omniprobe.utils.optim import cosine_decay_linear_warmup
from omniprobe.utils.progress import progress
from omniprobe.utils.training import ddp_setup, ddp_cleanup, unwrap_model


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    loss_fn,
    device,
    rank=0,
    world_size=1,
    valid_loader=None,
    scale_invariant=False,
    max_steps_per_epoch=None,
):
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0.0
        steps_this_epoch = 0
        pbar = progress(train_loader, desc=f"Epoch {ep}") if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            images = batch["image"].to(device)
            target = batch["depth"].to(device)

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
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            if scale_invariant:
                pred = match_scale_and_shift(pred, target)
                pred = pred.clamp(min=0.001, max=10.0)

            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) probe_lr: {pr_lr:.2e}"
                )

            steps_this_epoch += 1
            if (max_steps_per_epoch is not None) and (
                steps_this_epoch >= max_steps_per_epoch
            ):
                break

        denominator = (
            steps_this_epoch
            if steps_this_epoch > 0
            else len(train_loader)
        )
        train_loss /= denominator

        if rank == 0:
            logger.info(f"train loss {ep}   | {train_loss:.4f}")
            if valid_loader is not None:
                val_loss, val_metrics = validate(
                    model,
                    probe,
                    valid_loader,
                    loss_fn,
                    device,
                    scale_invariant=scale_invariant,
                )
                logger.info(f"valid loss {ep}   | {val_loss:.4f}")
                for metric in val_metrics:
                    logger.info(f"valid SA {metric:10s} | {val_metrics[metric]:.4f}")


def validate(
    model,
    probe,
    loader,
    loss_fn,
    device,
    verbose=True,
    scale_invariant=False,
    aggregate=True,
    max_batches=None,
):
    total_loss = 0.0
    metrics = None
    batches_processed = 0
    with torch.inference_mode():
        pbar = progress(loader, desc="Evaluation") if verbose else loader
        for batch in pbar:
            images = batch["image"].to(device)
            target = batch["depth"].to(device)

            feat = model(images)
            pred = probe(feat).detach()
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            loss = loss_fn(pred, target)
            total_loss += loss.item()

            batch_metrics = evaluate_depth(
                pred, target, scale_invariant=scale_invariant
            )
            if metrics is None:
                metrics = {key: [value] for key, value in batch_metrics.items()}
            else:
                for key, value in batch_metrics.items():
                    metrics[key].append(value)
            batches_processed += 1
            if (max_batches is not None) and (batches_processed >= max_batches):
                break

    # aggregate
    denominator = (
        batches_processed if batches_processed > 0 else len(loader)
    )
    total_loss = total_loss / denominator
    for key in metrics:
        metric_key = torch.cat(metrics[key], dim=0)
        metrics[key] = metric_key.mean() if aggregate else metric_key

    return total_loss, metrics


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    # ===== GET DATA LOADERS =====
    # validate and test on single gpu
    trainval_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    trainval_loader.dataset.__getitem__(0)

    # ===== Get models =====
    model = instantiate(cfg.backbone)
    probe = instantiate(
        cfg.probe, feat_dim=model.feat_dim, max_depth=trainval_loader.dataset.max_depth
    )

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
        f"{str(cfg.optimizer.probe_lr):>10s}",
        f"{str(cfg.optimizer.model_lr):>10s}",
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

    # move to DDP
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

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(  # noqa: E731
        epoch,
        cfg.optimizer.n_epochs * steps_per_epoch,
        int(cfg.optimizer.warmup_epochs * steps_per_epoch),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    loss_fn = DepthLoss()

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
        loss_fn=loss_fn,
        device=device,
        rank=rank,
        world_size=world_size,
        # valid_loader=test_loader,
        max_steps_per_epoch=max_steps_per_epoch,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        test_sa_loss, test_sa_metrics = validate(
            model,
            probe,
            test_loader,
            loss_fn,
            device,
            max_batches=max_eval_batches,
        )
        logger.info(f"Scale-Aware Final test loss       | {test_sa_loss:.4f}")
        for metric in test_sa_metrics:
            logger.info(f"Final test SA {metric:10s} | {test_sa_metrics[metric]:.4f}")
        results_sa = ", ".join([f"{test_sa_metrics[_m]:.4f}" for _m in test_sa_metrics])

        # get scale invariant
        test_si_loss, test_si_metrics = validate(
            model,
            probe,
            test_loader,
            loss_fn,
            device,
            scale_invariant=True,
            max_batches=max_eval_batches,
        )
        logger.info(f"Scale-Invariant Final test loss       | {test_si_loss:.4f}")
        for metric in test_si_metrics:
            logger.info(f"Final test SI {metric:10s} | {test_si_metrics[metric]:.4f}")
        results_si = ", ".join([f"{test_si_metrics[_m]:.4f}" for _m in test_si_metrics])

        entry = build_result_entry(
            "depth",
            model,
            output_dir,
            cfg,
            {
                **{
                    f"sa_{metric}": float(test_sa_metrics[metric].item())
                    for metric in test_sa_metrics
                },
                **{
                    f"si_{metric}": float(test_si_metrics[metric].item())
                    for metric in test_si_metrics
                },
            },
            probe="; ".join(probe_info),
            training="; ".join(train_info),
            test_dataset=test_dset,
        )
        append_jsonl(
            resolve_results_path(cfg, f"depth_{test_dset}.jsonl"),
            entry,
        )

        # save final model
        ckpt_path = artifact_dir(cfg, "checkpoints") / "ckpt.pth"
        model_state = unwrap_model(model).state_dict()
        probe_state = unwrap_model(probe).state_dict()
        checkpoint = {
            "cfg": cfg,
            "model": model_state,
            "probe": probe_state,
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
