
import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

if not hasattr(torch, "_six"):
    class _TorchSix:
        string_classes = (str,)

    torch._six = _TorchSix()


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


@dataclass
class ImageNetDataConfig:
    root: str
    split: str
    image_size: int = 224
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    mean: Tuple[float, float, float] = tuple(IMAGENET_DEFAULT_MEAN)
    std: Tuple[float, float, float] = tuple(IMAGENET_DEFAULT_STD)


def build_imagenet_transform(image_size: int, mean, std, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.2, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    resize_size = int((256 / 224) * image_size)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_imagenet_dataloader(
    cfg: ImageNetDataConfig,
    train: bool,
) -> DataLoader:
    split_path = os.path.join(cfg.root, cfg.split)
    transform = build_imagenet_transform(cfg.image_size, cfg.mean, cfg.std, train=train)
    dataset = datasets.ImageFolder(split_path, transform=transform)
    shuffle = train
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=train,
    )
    return loader
