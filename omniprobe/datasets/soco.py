"""Dataset loader for SOCO object correspondence pairs."""


import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class SOCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = "data/Object_Correspondence",
        image_size: int = 512,
        image_mean: str = "imagenet",
        use_bbox: bool = True,
        class_name: Optional[str] = None,
        max_pairs: Optional[int] = None,
        pair_subdir: str = "PairAnnotations",
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.image_root = self.root / "Images"
        pair_path = Path(pair_subdir)
        if not pair_path.is_absolute():
            pair_path = self.root / pair_subdir
        self.pair_root = pair_path
        self.use_bbox = use_bbox
        self.image_size = image_size

        if image_mean == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        elif image_mean == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif image_mean in ["raw", "zeros"]:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        else:
            raise ValueError(f"Unsupported image_mean '{image_mean}'")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )

        class_dirs = [class_name] if class_name else sorted(os.listdir(self.pair_root))
        instances: List[Dict] = []
        for cls in class_dirs:
            cls_dir = self.pair_root / cls
            if not cls_dir.exists():
                continue
            pair_files = sorted(glob.glob(str(cls_dir / "*.json")))
            if max_pairs:
                pair_files = pair_files[:max_pairs]
            for pf in pair_files:
                with open(pf, "r") as f:
                    payload = json.load(f)
                payload["pair_path"] = pf
                instances.append(payload)

        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def _load_image(self, class_name: str, image_name: str, bbox: Optional[Sequence[float]]) -> Image.Image:
        path = self.image_root / class_name / image_name
        with Image.open(path) as img:
            image = np.array(img.convert("RGB"))

        if bbox and self.use_bbox:
            h, w, _ = image.shape
            l, u, r, d = bbox
            l = int(max(0, np.floor(l)))
            u = int(max(0, np.floor(u)))
            r = int(min(w, np.ceil(r)))
            d = int(min(h, np.ceil(d)))
            image = image[u:d, l:r]

        h, w, _ = image.shape
        if h == 0 or w == 0:
            raise RuntimeError(f"Invalid crop for {class_name}/{image_name}")

        if h != w:
            max_hw = max(h, w)
            pad_h = max_hw - h
            pad_w = max_hw - w
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=255,
            )

        return Image.fromarray(image)

    @staticmethod
    def _keypoints_to_tensor(keypoints: List[Dict]) -> Tuple[torch.Tensor, Dict[str, int]]:
        names = [kp["name"] for kp in keypoints]
        mapping = {name: idx for idx, name in enumerate(names)}
        tensor = torch.zeros(len(keypoints), 3).int()
        for idx, kp in enumerate(keypoints):
            tensor[idx, 0] = int(kp["pos"][0])
            tensor[idx, 1] = int(kp["pos"][1])
            tensor[idx, 2] = 1
        return tensor, mapping

    def __getitem__(self, index: int):
        pair = self.instances[index]
        pair_group = pair.get("pair_group", pair.get("category", "unknown"))
        src_class = pair.get("src_category", pair.get("category"))
        trg_class = pair.get("trg_category", pair.get("category"))

        src_bbox = pair.get("src_bndbox")
        trg_bbox = pair.get("trg_bndbox")

        img_i = self._load_image(src_class, pair["src_image"], src_bbox)
        img_j = self._load_image(trg_class, pair["trg_image"], trg_bbox)

        kps_i, map_i = self._keypoints_to_tensor(pair["src_keypoints"])
        kps_j, map_j = self._keypoints_to_tensor(pair["trg_keypoints"])
        src_names = [kp["name"] for kp in pair["src_keypoints"]]
        trg_names = [kp["name"] for kp in pair["trg_keypoints"]]
        src_concepts = [kp.get("concept_name") for kp in pair["src_keypoints"]]
        trg_concepts = [kp.get("concept_name") for kp in pair["trg_keypoints"]]

        hw_i = img_i.size[0]
        hw_j = img_j.size[0]

        img_i = self.image_transform(img_i)
        img_j = self.image_transform(img_j)

        mask_i_img = Image.fromarray(np.ones((hw_i, hw_i), dtype=np.uint8) * 255)
        mask_j_img = Image.fromarray(np.ones((hw_j, hw_j), dtype=np.uint8) * 255)
        mask_i = self.mask_transform(mask_i_img)
        mask_j = self.mask_transform(mask_j_img)

        kps_i[:, :2] = kps_i[:, :2] * self.image_size / hw_i
        kps_j[:, :2] = kps_j[:, :2] * self.image_size / hw_j

        if self.use_bbox:
            thresh_scale = 1.0
        else:
            if trg_bbox:
                l, u, r, d = trg_bbox
                max_bbox = max(r - l, d - u)
                max_idim = max(pair["trg_imsize"][:2])
                thresh_scale = float(max_bbox) / max_idim
            else:
                thresh_scale = 1.0

        semantic_pairs = []
        for name in pair.get("semantic_overlap", []):
            if name in map_i and name in map_j:
                semantic_pairs.append((map_i[name], map_j[name]))

        concept_map: Dict[int, set] = {}
        for concept in pair.get("concept_matches", []):
            trg_indices = [map_j[name] for name in concept.get("trg_keypoints", []) if name in map_j]
            if not trg_indices:
                continue
            for src_name in concept.get("src_keypoints", []):
                if src_name not in map_i:
                    continue
                idx = map_i[src_name]
                concept_map.setdefault(idx, set()).update(trg_indices)

        concept_map = {k: sorted(list(v)) for k, v in concept_map.items() if v}

        meta = {
            "semantic_pairs": semantic_pairs,
            "concept_map": concept_map,
            "pair_id": pair["pair_id"],
            "class_name": pair_group,
            "pair_group": pair_group,
            "src_class": src_class,
            "trg_class": trg_class,
            "src_name": pair["src_image"],
            "trg_name": pair["trg_image"],
            "pair_path": pair.get("pair_path"),
            "src_names": src_names,
            "trg_names": trg_names,
            "src_concepts": src_concepts,
            "trg_concepts": trg_concepts,
            "concept_matches": pair.get("concept_matches", []),
        }

        return img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, meta