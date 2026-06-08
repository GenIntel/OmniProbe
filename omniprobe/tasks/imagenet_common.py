from omniprobe.datasets.imagenet import ImageNetDataConfig, build_imagenet_dataloader
from omniprobe.utils.eval_helpers import resolve_mean_std


def build_imagenet_loaders(task_cfg, backbone_cfg):
    mean, std = resolve_mean_std(backbone_cfg.image_mean)
    train_cfg = ImageNetDataConfig(
        root=task_cfg.data_root,
        split=task_cfg.train_split,
        image_size=task_cfg.image_size,
        batch_size=task_cfg.batch_size,
        num_workers=task_cfg.num_workers,
        mean=tuple(mean),
        std=tuple(std),
    )
    val_cfg = ImageNetDataConfig(
        root=task_cfg.data_root,
        split=task_cfg.val_split,
        image_size=task_cfg.image_size,
        batch_size=task_cfg.batch_size,
        num_workers=task_cfg.num_workers,
        mean=tuple(mean),
        std=tuple(std),
        pin_memory=False,
    )
    return (
        build_imagenet_dataloader(train_cfg, train=True),
        build_imagenet_dataloader(val_cfg, train=False),
    )
