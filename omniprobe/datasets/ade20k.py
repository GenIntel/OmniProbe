"""
ADE20K dataset utilities for linear segmentation probes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from omniprobe.utils.eval_helpers import resolve_mean_std
from omniprobe.utils.paths import require_env_path

# ---------------------------------------------------------------------------
# Pair transforms (adapted from NeCo linear_finetuning_transforms.py)
# ---------------------------------------------------------------------------


class RandomHorizontalFlipPair:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if torch.rand(1).item() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomResizedCropPair:
    def __init__(self, size: int, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.transform = T.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    def __call__(self, image: Image.Image, mask: Image.Image):
        top, left, height, width = T.RandomResizedCrop.get_params(
            image, self.transform.scale, self.transform.ratio
        )
        image = F.resized_crop(
            image,
            top,
            left,
            height,
            width,
            self.transform.size,
            InterpolationMode.BILINEAR,
        )
        mask = F.resized_crop(
            mask,
            top,
            left,
            height,
            width,
            self.transform.size,
            InterpolationMode.NEAREST,
        )
        return image, mask


class ToTensorPair:
    def __call__(self, image: Image.Image, mask: Image.Image):
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


class NormalizePair:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask


class ComposePair:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class SeparateTransforms:
    def __init__(self, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask):
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return image, mask


def default_train_transforms(size: int, image_mean: str = "imagenet") -> ComposePair:
    mean, std = resolve_mean_std(image_mean)
    return ComposePair(
        [
            RandomResizedCropPair(size=size, scale=(0.8, 1.0)),
            RandomHorizontalFlipPair(p=0.5),
            ToTensorPair(),
            NormalizePair(mean=mean, std=std),
        ]
    )


def default_val_transforms(size: int, image_mean: str = "imagenet") -> SeparateTransforms:
    mean, std = resolve_mean_std(image_mean)
    img_transform = T.Compose(
        [
            T.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    mask_transform = T.Compose(
        [
            T.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            lambda mask: torch.from_numpy(np.array(mask, dtype=np.int64)),
        ]
    )
    return SeparateTransforms(img_transform, mask_transform)


# ---------------------------------------------------------------------------
# Dataset and dataloader helpers
# ---------------------------------------------------------------------------


class ADE20KDataset(Dataset):
    split_to_dir = {"train": "training", "val": "validation"}

    def __init__(
        self,
        root: str,
        split: str,
        transforms: Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]],
        file_set: Optional[list[str]] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.file_set = file_set
        self.samples = self._collect_files()

    def _collect_files(self):
        image_dir = self.root / "images" / self.split_to_dir[self.split]
        mask_dir = self.root / "annotations" / self.split_to_dir[self.split]
        if self.file_set is None:
            image_paths = sorted(image_dir.glob("*.jpg"))
            mask_paths = sorted(mask_dir.glob("*.png"))
        else:
            image_paths = [image_dir / f"{name}.jpg" for name in sorted(self.file_set)]
            mask_paths = [mask_dir / f"{name}.png" for name in sorted(self.file_set)]
        assert len(image_paths) == len(
            mask_paths
        ), "Image and mask counts mismatch in ADE20K."
        return list(zip(image_paths, mask_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = self.transforms(image, mask)
        return {"image": image, "mask": mask}


@dataclass
class ADE20KDataConfig:
    root: Optional[str] = None
    crop_size: int = 448
    batch_size: int = 128
    val_batch_size: Optional[int] = None
    num_workers: int = 8
    shuffle: bool = True
    val_iters: Optional[int] = 512
    train_mask_size: int = 100
    val_mask_size: int = 100
    ignore_index: int = 0
    num_classes: int = 151
    image_mean: str = "imagenet"


def build_ade20k_dataloaders(
    cfg: ADE20KDataConfig,
) -> Tuple[DataLoader, DataLoader]:
    root = cfg.root or require_env_path("ADE20K_ROOT", "ADE20K dataset root")
    train_transforms = default_train_transforms(cfg.crop_size, image_mean=cfg.image_mean)
    val_transforms = default_val_transforms(cfg.crop_size, image_mean=cfg.image_mean)

    train_dataset = ADE20KDataset(
        root=root,
        split="train",
        transforms=train_transforms,
    )
    val_dataset = ADE20KDataset(
        root=root,
        split="val",
        transforms=val_transforms,
    )

    val_batch_size = cfg.val_batch_size or cfg.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
