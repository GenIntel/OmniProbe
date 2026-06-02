
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from omniprobe.utils.pose import continuous_to_bin
from omniprobe.utils.eval_helpers import resolve_mean_std


class ImageNet3DPoseDataset(Dataset):
    """
    Dataset that serves ImageNet3D object-centric crops together with pose
    annotations. Assumes data has been preprocessed with
    ``scripts/preprocess_data.py`` from ``imagenet3d_exp`` and stored under
    ``root/split/(images|annotations|lists)``.
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 224,
        num_bins: int = 40,
        min_angle: float = 0.0,
        max_angle: float = 2 * math.pi,
        image_mean: str | None = None,
        mean_std: str = "imagenet",
        augment: bool = True,
        categories: Sequence[str] | None = None,
    ):
        super().__init__()
        assert split in {"train", "val"}, f"Unsupported split: {split}"
        self.name = "imagenet3d_pose"
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.num_bins = num_bins
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.augment = augment and split == "train"

        # `mean_std` is kept for backward compatibility with existing configs.
        if image_mean is None:
            image_mean = mean_std
        mean, std = resolve_mean_std(image_mean)

        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

        split_root = self.root / split
        image_dir = split_root / "images"
        list_dir = split_root / "lists"
        annot_dir = split_root / "annotations"
        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory '{image_dir}' does not exist. "
                "Please preprocess ImageNet3D and update configs/dataset/imagenet3d_pose.yaml."
            )

        if categories is None:
            categories = sorted([p.name for p in image_dir.iterdir() if p.is_dir()])
        else:
            categories = sorted(categories)

        self.categories: List[str] = categories
        self.category_to_index: Dict[str, int] = {c: i for i, c in enumerate(categories)}

        self.samples: List[Dict] = []
        for cate in self.categories:
            mesh_list = list_dir / cate / "mesh01.txt"
            if not mesh_list.exists():
                raise FileNotFoundError(f"Missing list file '{mesh_list}' for category '{cate}'")
            with open(mesh_list, "r") as f:
                names = [line.strip() for line in f.readlines() if line.strip()]
            for name in names:
                self.samples.append(
                    {
                        "category": cate,
                        "category_idx": self.category_to_index[cate],
                        "name": name,
                        "image_path": image_dir / cate / f"{name}.JPEG",
                        "annot_path": annot_dir / cate / f"{name}.npz",
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in '{split_root}'. Check preprocessing output.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.resize(img)
            img = self.to_tensor(img)
        return img

    def _maybe_flip(self, image: torch.Tensor, azimuth: float, theta: float):
        if not self.augment or random.random() > 0.5:
            return image, azimuth, theta
        image = TF.hflip(image)
        azimuth = (2 * math.pi - azimuth) % (2 * math.pi)
        theta = -theta
        return image, azimuth, theta

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        image = self._load_image(sample["image_path"])
        annot = dict(np.load(sample["annot_path"], allow_pickle=True))
        if "annotations" in annot:
            raise ValueError(
                "Expected single-object annotations. "
                f"Please preprocess ImageNet3D with center cropping (see imagenet3d_exp). Problematic file: {sample['annot_path']}"
            )

        azimuth = float(annot["azimuth"])
        elevation = float(annot["elevation"])
        theta = float(annot["theta"])

        image = self.normalize(image)
        image, azimuth, theta = self._maybe_flip(image, azimuth, theta)

        az_idx = continuous_to_bin(
            azimuth,
            num_bins=self.num_bins,
            min_value=self.min_angle,
            max_value=self.max_angle,
        )
        el_idx = continuous_to_bin(
            elevation,
            num_bins=self.num_bins,
            min_value=self.min_angle,
            max_value=self.max_angle,
        )
        th_idx = continuous_to_bin(
            theta,
            num_bins=self.num_bins,
            min_value=self.min_angle,
            max_value=self.max_angle,
        )

        return {
            "image": image,
            "azimuth": torch.tensor(azimuth, dtype=torch.float32),
            "elevation": torch.tensor(elevation, dtype=torch.float32),
            "theta": torch.tensor(theta, dtype=torch.float32),
            "azimuth_idx": torch.tensor(az_idx, dtype=torch.long),
            "elevation_idx": torch.tensor(el_idx, dtype=torch.long),
            "theta_idx": torch.tensor(th_idx, dtype=torch.long),
            "category": sample["category_idx"],
            "category_name": sample["category"],
            "name": sample["name"],
        }
