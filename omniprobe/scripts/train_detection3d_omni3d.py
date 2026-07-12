"""Cube R-CNN 3D object detection on Omni3D datasets (default: ARKitScenes).

Trains a Cube R-CNN detector on top of a (typically frozen) OmniProbe
backbone: the backbone's four dense feature maps feed a DPT_FPN pyramid
probe, and the vendored Cube R-CNN heads predict 2D boxes and 3D cuboids.
Reports AP2D and AP3D (plus threshold/range breakdowns).

Requires the optional detection stack: detectron2, pytorch3d, and the
``detection3d`` extra.

Adapted from the omni3d training loop (tools/train_net.py in
https://github.com/facebookresearch/omni3d, CC-BY-NC 4.0).
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from omniprobe.runtime import (
    append_jsonl,
    build_result_entry,
    resolve_results_path,
)
from omniprobe.utils.eval_helpers import resolve_mean_std

logger = logging.getLogger("cubercnn")

# rolling-loss divergence tolerance (see do_train)
LOSS_TOLERANCE = 4.0
LOSS_GAMMA = 0.02


def _vendor_base_config_path() -> Path:
    import omniprobe.models.vendor.cubercnn as cubercnn_pkg

    return Path(cubercnn_pkg.__file__).parent / "configs" / "omni3d_base.yaml"


def resolve_category_names(category_names):
    """Resolves a category preset name into the corresponding category list.

    Accepts either an explicit list of category names or one of the preset
    names understood by ``get_omni3d_categories`` (``omni3d``, ``omni3d_in``,
    ``omni3d_out``, or a registered split name such as ``KITTI_test``).
    """
    if isinstance(category_names, str):
        from omniprobe.models.vendor.cubercnn.data.builtin import (
            get_omni3d_categories,
        )

        return sorted(get_omni3d_categories(category_names))
    return list(category_names)


def build_d2_cfg(cfg):
    """Builds the frozen detectron2 CfgNode from the Hydra task config."""
    from detectron2.config import get_cfg

    from omniprobe.models.vendor.cubercnn.config import get_cfg_defaults

    d2 = get_cfg()
    get_cfg_defaults(d2)
    d2.merge_from_file(str(_vendor_base_config_path()))

    category_names = resolve_category_names(cfg.datasets.category_names)
    num_classes = int(cfg.datasets.get("num_classes") or len(category_names))

    d2.DATASETS.TRAIN = tuple(cfg.datasets.train)
    d2.DATASETS.TEST = tuple(cfg.datasets.test)
    d2.DATASETS.CATEGORY_NAMES = category_names
    d2.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    d2.SOLVER.TYPE = str(cfg.solver.type)
    d2.SOLVER.BASE_LR = float(cfg.solver.base_lr)
    d2.SOLVER.IMS_PER_BATCH = int(cfg.solver.ims_per_batch)
    d2.SOLVER.MAX_ITER = int(cfg.solver.max_iter)
    d2.SOLVER.STEPS = tuple(int(s) for s in cfg.solver.steps)
    d2.SOLVER.WARMUP_ITERS = int(cfg.solver.warmup_iters)
    d2.SOLVER.CHECKPOINT_PERIOD = int(cfg.solver.checkpoint_period)
    d2.SOLVER.AMP.ENABLED = bool(cfg.solver.amp)
    d2.TEST.EVAL_PERIOD = int(cfg.test.eval_period)

    d2.MODEL.STABILIZE = float(cfg.stabilize)
    d2.MODEL.WEIGHTS = str(cfg.weights)
    d2.OUTPUT_DIR = str(cfg.output_dir)
    d2.SEED = int(cfg.system.random_seed)
    d2.TEST.VISUALIZE_PREDICTIONS = bool(cfg.get("visualize_predictions", False))

    if str(cfg.pixel_norm) == "backbone":
        # normalize with the backbone's own preset instead of the caffe stats
        mean, std = resolve_mean_std(cfg.image_mean)
        d2.MODEL.PIXEL_MEAN = [m * 255.0 for m in mean]
        d2.MODEL.PIXEL_STD = [s * 255.0 for s in std]

    overrides = list(cfg.get("d2_overrides", []) or [])
    if overrides:
        d2.merge_from_list([str(item) for item in overrides])

    # visualization was removed from the vendored RCNN3D; force it off so an
    # override cannot silently request a feature that no longer exists
    if d2.VIS_PERIOD != 0:
        logger.warning("VIS_PERIOD is not supported by this task; forcing 0.")
    d2.VIS_PERIOD = 0

    d2.freeze()
    return d2


def build_backbone_adapter(cfg, device):
    """Instantiates the OmniProbe backbone + pyramid probe as a d2 backbone."""
    from hydra.utils import instantiate

    from omniprobe.models.detectron2_backbone import OmniProbeD2Backbone

    model = instantiate(cfg.backbone)
    feat_dims = (
        list(model.feat_dim)
        if isinstance(model.feat_dim, (list, tuple))
        else [model.feat_dim] * 4
    )
    probe = instantiate(cfg.probe, input_dims=feat_dims)
    adapter = OmniProbeD2Backbone(model, probe, freeze=bool(cfg.freeze_backbone))
    return adapter.to(torch.device(device))


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "Trainable Parameters": trainable,
        "Frozen Parameters": frozen,
        "Total Parameters": trainable + frozen,
    }


def allreduce_dict(input_dict, average=True):
    """Reduces dict values across processes; rank 0 holds the result."""
    import torch.distributed as dist
    from detectron2.utils import comm

    world_size = comm.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in names], dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(names, values)}


def do_test(d2_cfg, model, iteration="final"):
    """Runs inference + AP2D/AP3D evaluation; returns the evaluation helper."""
    from detectron2.utils import comm

    from omniprobe.models.vendor.cubercnn.data import (
        build_detection_test_loader,
        get_filter_settings_from_cfg,
    )
    from omniprobe.models.vendor.cubercnn.evaluation import (
        Omni3DEvaluationHelper,
        inference_on_dataset,
    )

    filter_settings = get_filter_settings_from_cfg(d2_cfg)
    filter_settings["visibility_thres"] = d2_cfg.TEST.VISIBILITY_THRES
    filter_settings["truncation_thres"] = d2_cfg.TEST.TRUNCATION_THRES
    filter_settings["min_height_thres"] = 0.0625
    filter_settings["max_depth"] = 1e8

    dataset_names_test = d2_cfg.DATASETS.TEST
    only_2d = d2_cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(
        d2_cfg.OUTPUT_DIR, "inference", "iter_{}".format(iteration)
    )

    eval_helper = Omni3DEvaluationHelper(
        dataset_names_test,
        filter_settings,
        output_folder,
        iter_label=iteration,
        only_2d=only_2d,
    )

    for dataset_name in dataset_names_test:
        data_loader = build_detection_test_loader(d2_cfg, dataset_name)
        results_json = inference_on_dataset(model, data_loader)

        if comm.is_main_process():
            eval_helper.add_predictions(dataset_name, results_json)
            eval_helper.save_predictions(dataset_name)
            eval_helper.evaluate(dataset_name)

            if d2_cfg.TEST.VISUALIZE_PREDICTIONS:
                # render qualitative 3D cuboid predictions (every 50th image)
                # into <output_folder>/<dataset>/vis/, as upstream Cube R-CNN
                from detectron2.data import MetadataCatalog

                from omniprobe.models.vendor.cubercnn.vis import vis

                instances = torch.load(
                    os.path.join(
                        output_folder, dataset_name, "instances_predictions.pth"
                    ),
                    weights_only=False,
                )
                log_str = vis.visualize_from_instances(
                    instances,
                    data_loader.dataset,
                    dataset_name,
                    d2_cfg.INPUT.MIN_SIZE_TEST,
                    os.path.join(output_folder, dataset_name),
                    MetadataCatalog.get("omni3d_model").thing_classes,
                    iteration,
                )
                logger.info(log_str)

    if comm.is_main_process():
        eval_helper.summarize_all()

    return eval_helper


def do_train(d2_cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=False):
    """One training attempt; returns False if divergence forces a retry."""
    import torch.distributed as dist
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.engine import default_writers
    from detectron2.solver import build_lr_scheduler
    from detectron2.utils import comm
    from detectron2.utils.events import EventStorage

    from omniprobe.models.vendor.cubercnn.data import (
        DatasetMapper3D,
        build_detection_train_loader,
    )
    from omniprobe.models.vendor.cubercnn.solver import (
        PeriodicCheckpointerOnlyOne,
        build_optimizer,
        freeze_bn,
    )

    max_iter = d2_cfg.SOLVER.MAX_ITER
    do_eval = d2_cfg.TEST.EVAL_PERIOD > 0
    use_amp = d2_cfg.SOLVER.AMP.ENABLED
    stabilize = float(d2_cfg.MODEL.STABILIZE)
    stabilizer_enabled = stabilize > 0

    model.train()
    device = next(model.parameters()).device

    optimizer = build_optimizer(d2_cfg, model)
    scheduler = build_lr_scheduler(d2_cfg, optimizer)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and torch.cuda.is_available()
    )

    checkpointer = DetectionCheckpointer(
        model, d2_cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    periodic_checkpointer = PeriodicCheckpointerOnlyOne(
        checkpointer, d2_cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = (
        default_writers(d2_cfg.OUTPUT_DIR, max_iter)
        if comm.is_main_process()
        else []
    )

    data_mapper = DatasetMapper3D(d2_cfg, is_train=True)
    data_loader = build_detection_train_loader(
        d2_cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src
    )
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats

    if d2_cfg.MODEL.WEIGHTS_PRETRAIN != "":
        # load ONLY the model, no checkpointables
        checkpointer.load(d2_cfg.MODEL.WEIGHTS_PRETRAIN, checkpointables=[])

    start_iter = (
        checkpointer.resume_or_load(d2_cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))

    if not d2_cfg.MODEL.USE_BN:
        freeze_bn(model)

    world_size = comm.get_world_size()

    # If the loss diverges for more than STABILIZE (as a fraction of
    # iterations), this attempt is abandoned and the caller starts a fresh
    # attempt that resumes from the latest checkpoint. Individual bad
    # updates are skipped.
    iterations_success = 0
    iterations_explode = 0
    recent_loss = None

    data_iter = iter(data_loader)
    named_params = list(model.named_parameters())

    with EventStorage(start_iter) as storage:

        while True:

            data = next(data_iter)
            storage.iter = iteration

            with torch.autocast("cuda", enabled=use_amp):
                loss_dict = model(data)
                losses = sum(loss_dict.values())

            loss_dict_reduced = {
                k: v.item() for k, v in allreduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            comm.synchronize()

            if recent_loss is None:
                recent_loss = losses_reduced * 2.0

            diverging_model = stabilizer_enabled and (
                losses_reduced > recent_loss * LOSS_TOLERANCE
                or not np.isfinite(losses_reduced)
                or np.isnan(losses_reduced)
            )

            if diverging_model:
                losses = losses.clip(0, 1)
                logger.warning(
                    "Skipping gradient update due to higher than normal loss {:.2f} "
                    "vs. rolling mean {:.2f}, Dict-> {}".format(
                        losses_reduced, recent_loss, loss_dict_reduced
                    )
                )
            else:
                recent_loss = (
                    recent_loss * (1 - LOSS_GAMMA) + losses_reduced * LOSS_GAMMA
                )

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            scaler.scale(losses).backward()

            grads_unscaled = False
            if not diverging_model and stabilizer_enabled:
                if use_amp:
                    scaler.unscale_(optimizer)
                    grads_unscaled = True
                for name, param in named_params:
                    if param.grad is not None:
                        diverging_model = (
                            torch.isnan(param.grad).any()
                            or torch.isinf(param.grad).any()
                        )
                    if diverging_model:
                        logger.warning(
                            "Skipping gradient update due to inf/nan detection, "
                            "loss is {}".format(loss_dict_reduced)
                        )
                        break

            # if any process detected divergence, all processes skip together
            diverging_model = torch.tensor(float(diverging_model), device=device)
            if world_size > 1:
                dist.all_reduce(diverging_model)
            comm.synchronize()

            if diverging_model > 0:
                optimizer.zero_grad()
                if use_amp:
                    # update() needs inf checks recorded for this step; the
                    # loss-spike path skips the gradient scan, so record them
                    if not grads_unscaled:
                        scaler.unscale_(optimizer)
                    scaler.update()
                iterations_explode += 1
            else:
                scaler.step(optimizer)
                scaler.update()
                storage.put_scalar(
                    "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
                )
                iterations_success += 1

            total_iterations = iterations_success + iterations_explode

            # only retry once sufficiently far past the latest checkpoint;
            # a disabled stabilizer disables the retry machinery entirely
            retry = (
                stabilizer_enabled
                and (iterations_explode / total_iterations) >= stabilize
                and total_iterations > d2_cfg.SOLVER.CHECKPOINT_PERIOD * 1 / 2
            )
            retry = torch.tensor(float(retry), device=device)
            if world_size > 1:
                dist.all_reduce(retry)
            comm.synchronize()

            if retry > 0:
                logger.warning(
                    "!! Restarting training at {} iters. Exploding loss {:d}% of iters !!".format(
                        iteration,
                        int(100 * (iterations_explode / total_iterations)),
                    )
                )
                del data_mapper
                del data_loader
                del optimizer
                del checkpointer
                del periodic_checkpointer
                return False

            scheduler.step()

            if not (diverging_model > 0) and (
                do_eval
                and ((iteration + 1) % d2_cfg.TEST.EVAL_PERIOD) == 0
                and iteration != (max_iter - 1)
            ):
                logger.info("Starting test for iteration {}".format(iteration + 1))
                do_test(d2_cfg, model, iteration=iteration + 1)
                comm.synchronize()
                model.train()

                if not d2_cfg.MODEL.USE_BN:
                    freeze_bn(model)

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()

            # skip checkpointing while the model looks like it may diverge;
            # with the stabilizer off, always checkpoint
            if not (diverging_model > 0) and (
                not stabilizer_enabled
                or (iterations_explode / total_iterations) < 0.5 * stabilize
            ):
                periodic_checkpointer.step(iteration)

            iteration += 1

            if iteration >= max_iter:
                break

    return True


def _register_datasets(cfg, d2_cfg):
    from omniprobe.models.vendor.cubercnn.data import (
        get_filter_settings_from_cfg,
        simple_register,
    )

    filter_settings = get_filter_settings_from_cfg(d2_cfg)
    dataset_root = str(cfg.dataset_root)

    for dataset_name in d2_cfg.DATASETS.TRAIN:
        simple_register(
            dataset_name,
            filter_settings,
            filter_empty=True,
            datasets_root_path=dataset_root,
        )
    for dataset_name in d2_cfg.DATASETS.TEST:
        if dataset_name not in d2_cfg.DATASETS.TRAIN:
            simple_register(
                dataset_name,
                filter_settings,
                filter_empty=False,
                datasets_root_path=dataset_root,
            )
    return filter_settings


def _prepare_metadata_and_priors(cfg, d2_cfg, filter_settings):
    """Registers category metadata; computes dimension priors for training."""
    from detectron2.data import MetadataCatalog

    from omniprobe.models.vendor.cubercnn import util
    from omniprobe.models.vendor.cubercnn.data import datasets as vendor_datasets

    dataset_root = str(cfg.dataset_root)

    if cfg.eval_only:
        # Load category metadata from an explicit file, from a reused run
        # dir, or (fresh run dir) derive it from the configured category
        # names and the dataset stats — register_and_store_model_metadata
        # covers all three and registers the omni3d_model metadata.
        explicit_path = str(cfg.get("category_meta_path", "") or "")
        if explicit_path:
            if not os.path.exists(explicit_path):
                raise FileNotFoundError(
                    f"category_meta_path does not exist: {explicit_path}"
                )
            meta_dir = os.path.dirname(explicit_path)
        else:
            meta_dir = str(cfg.output_dir)
        vendor_datasets.register_and_store_model_metadata(
            None, meta_dir, filter_settings, datasets_root_path=dataset_root
        )
        # priors are restored from the checkpoint state dict
        return None, None, None

    dataset_paths = [
        os.path.join(dataset_root, "Omni3D", name + ".json")
        for name in d2_cfg.DATASETS.TRAIN
    ]
    datasets = vendor_datasets.Omni3D(dataset_paths, filter_settings=filter_settings)

    vendor_datasets.register_and_store_model_metadata(
        datasets, d2_cfg.OUTPUT_DIR, filter_settings, datasets_root_path=dataset_root
    )

    thing_classes = MetadataCatalog.get("omni3d_model").thing_classes
    dataset_id_to_contiguous_id = MetadataCatalog.get(
        "omni3d_model"
    ).thing_dataset_id_to_contiguous_id

    infos = datasets.dataset["info"]
    if type(infos) == dict:
        infos = [datasets.dataset["info"]]

    dataset_id_to_unknown_cats = {}
    possible_categories = set(
        i for i in range(d2_cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1)
    )
    dataset_id_to_src = {}

    for info in infos:
        dataset_id = info["id"]
        known_category_training_ids = set()

        if dataset_id not in dataset_id_to_src:
            dataset_id_to_src[dataset_id] = info["source"]

        for cat_id in info["known_category_ids"]:
            if cat_id in dataset_id_to_contiguous_id:
                known_category_training_ids.add(dataset_id_to_contiguous_id[cat_id])

        dataset_id_to_unknown_cats[dataset_id] = (
            possible_categories - known_category_training_ids
        )

        logger.info("Available categories for {}".format(info["name"]))
        logger.info(
            [
                thing_classes[i]
                for i in (possible_categories & known_category_training_ids)
            ]
        )

    priors = util.compute_priors(d2_cfg, datasets)
    return priors, dataset_id_to_unknown_cats, dataset_id_to_src


def _flatten_eval_metrics(eval_helper, dataset_names_test):
    metrics = {}
    for dataset_name, res in eval_helper.results_analysis.items():
        single_test_set = len(dataset_names_test) == 1
        if single_test_set and dataset_name == "<Concat>":
            continue
        prefix = "" if single_test_set else f"{dataset_name}_"
        for key in (
            "AP2D",
            "AP3D",
            "AP3D@15",
            "AP3D@25",
            "AP3D@50",
            "AP3D@70",
            "AP3D-N",
            "AP3D-M",
            "AP3D-F",
        ):
            if key not in res:
                continue
            value = res[key]
            metrics[f"{prefix}{key}"] = (
                float(value) if np.isfinite(value) else None
            )
    return metrics


def _format_ap(value):
    try:
        if value is None or not np.isfinite(value):
            return "nan"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "nan"


def _write_summary(cfg, d2_cfg, backbone_model, eval_helper, metrics):
    """Writes a compact human-readable results file into the run directory."""
    lines = [
        f"task: {cfg.task_name}",
        f"backbone: {getattr(backbone_model, 'checkpoint_name', '?')} "
        f"(patch {getattr(backbone_model, 'patch_size', '?')}, "
        f"frozen={bool(cfg.freeze_backbone)})",
        f"training: {d2_cfg.SOLVER.MAX_ITER} iters, batch "
        f"{d2_cfg.SOLVER.IMS_PER_BATCH}, {d2_cfg.SOLVER.TYPE} "
        f"lr {d2_cfg.SOLVER.BASE_LR}",
        f"test datasets: {', '.join(d2_cfg.DATASETS.TEST)}",
        "",
        "overall:",
    ]
    for key, value in metrics.items():
        lines.append(f"  {key:<24} {_format_ap(value):>7}")

    for dataset_name, res in eval_helper.results.items():
        cats_2d = {
            key[3:]: value
            for key, value in res.get("bbox_2D", {}).items()
            if key.startswith("AP-")
        }
        cats_3d = {
            key[3:]: value
            for key, value in res.get("bbox_3D", {}).items()
            if key.startswith("AP-")
        }
        if not cats_2d:
            continue
        lines += ["", f"per-category ({dataset_name}):"]
        lines.append(f"  {'category':<20} {'AP2D':>7} {'AP3D':>7}")
        for cat in sorted(cats_2d):
            lines.append(
                f"  {cat:<20} {_format_ap(cats_2d[cat]):>7} "
                f"{_format_ap(cats_3d.get(cat)):>7}"
            )

    summary = "\n".join(lines) + "\n"
    summary_path = os.path.join(str(cfg.output_dir), "results_summary.txt")
    with open(summary_path, "w") as handle:
        handle.write(summary)
    logger.info("Results summary:\n{}".format(summary))
    logger.info("Wrote results summary to {}".format(summary_path))


def _write_results(cfg, d2_cfg, adapter, eval_helper):
    backbone_model = adapter.model
    metrics = _flatten_eval_metrics(eval_helper, d2_cfg.DATASETS.TEST)
    _write_summary(cfg, d2_cfg, backbone_model, eval_helper, metrics)
    probe_name = getattr(adapter.probe, "name", type(adapter.probe).__name__)
    entry = build_result_entry(
        str(cfg.task_name),
        backbone_model,
        cfg.output_dir,
        cfg,
        metrics,
        probe=probe_name,
        training=(
            "eval_only"
            if cfg.eval_only
            else (
                f"{d2_cfg.SOLVER.MAX_ITER}it; bs{d2_cfg.SOLVER.IMS_PER_BATCH}; "
                f"{d2_cfg.SOLVER.TYPE} {d2_cfg.SOLVER.BASE_LR}; "
                f"freeze={bool(cfg.freeze_backbone)}"
            )
        ),
        test_dataset=",".join(d2_cfg.DATASETS.TEST),
    )
    results_path = resolve_results_path(cfg, f"{cfg.task_name}.jsonl")
    append_jsonl(results_path, entry)
    logger.info("Appended results to {}".format(results_path))


def _main(payload):
    from detectron2.utils import comm
    from detectron2.utils.env import seed_all_rng
    from detectron2.utils.logger import setup_logger
    from torch.nn.parallel import DistributedDataParallel

    from omniprobe.models.detectron2_backbone import build_rcnn3d_model

    cfg = OmegaConf.create(payload)
    d2_cfg = build_d2_cfg(cfg)

    os.makedirs(d2_cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(
        output=d2_cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn"
    )
    # the vendored evaluator logs under omniprobe.models.vendor.cubercnn.*;
    # without a handler here its output (AP tables) is lost in spawned workers
    setup_logger(
        output=d2_cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="omniprobe"
    )
    seed_all_rng(d2_cfg.SEED + comm.get_rank())

    if comm.is_main_process():
        with open(os.path.join(d2_cfg.OUTPUT_DIR, "d2_config.yaml"), "w") as handle:
            handle.write(d2_cfg.dump())

    filter_settings = _register_datasets(cfg, d2_cfg)
    priors, dataset_id_to_unknown_cats, dataset_id_to_src = (
        _prepare_metadata_and_priors(cfg, d2_cfg, filter_settings)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    remaining_attempts = int(cfg.max_attempts)
    first_attempt = True
    while remaining_attempts > 0:

        backbone = build_backbone_adapter(cfg, device)
        model = build_rcnn3d_model(d2_cfg, backbone, priors, device)

        for key, value in count_parameters(model).items():
            logger.info(f"{key}: {value:,}")

        if first_attempt:
            logger.info("Model:\n{}".format(model))
            first_attempt = False

        if cfg.eval_only:
            from detectron2.checkpoint import DetectionCheckpointer

            DetectionCheckpointer(model, save_dir=d2_cfg.OUTPUT_DIR).resume_or_load(
                d2_cfg.MODEL.WEIGHTS, resume=bool(cfg.resume)
            )
            eval_helper = do_test(d2_cfg, model)
            if comm.is_main_process():
                _write_results(cfg, d2_cfg, backbone, eval_helper)
            return

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        # retries resume from the latest checkpoint saved by the failed attempt
        retrying = remaining_attempts < int(cfg.max_attempts)
        if do_train(
            d2_cfg,
            model,
            dataset_id_to_unknown_cats,
            dataset_id_to_src,
            resume=bool(cfg.resume) or retrying,
        ):
            break

        remaining_attempts -= 1
        del model
        del backbone

    if remaining_attempts == 0:
        raise ValueError("Training failed: model diverged in every attempt.")

    eval_helper = do_test(d2_cfg, model)
    if comm.is_main_process():
        _write_results(cfg, d2_cfg, backbone, eval_helper)


def run_task(cfg: DictConfig):
    from detectron2.engine import launch

    payload = OmegaConf.to_container(cfg, resolve=True)
    num_gpus = int(cfg.system.num_gpus)
    launch(
        _main,
        num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=f"tcp://127.0.0.1:{int(cfg.system.port)}",
        args=(payload,),
    )
