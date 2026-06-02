# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AP-10K Dataset for correspondence evaluation
# Adapted for omniprobe framework

import glob
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def get_ap10k_categories(data_root, eval_subset='intra-species', split='test'):
    """
    Get categories for AP-10K dataset based on evaluation subset.
    
    Args:
        data_root: Path to AP-10K dataset
        eval_subset: One of 'intra-species', 'cross-species', 'cross-family'
        split: Dataset split ('test', 'val', 'trn')
    
    Returns:
        categories: List of category names
        modified_split: Split name (may be modified for cross-species/cross-family)
    """
    subfolders = os.listdir(os.path.join(data_root, 'ImageAnnotation'))
    modified_split = split
    
    if eval_subset == 'intra-species':
        categories = [
            folder 
            for subfolder in subfolders 
            for folder in os.listdir(os.path.join(data_root, 'ImageAnnotation', subfolder))
        ]
    elif eval_subset == 'cross-species':
        categories = [
            subfolder 
            for subfolder in subfolders 
            if len(os.listdir(os.path.join(data_root, 'ImageAnnotation', subfolder))) > 1
        ]
        modified_split = split + '_cross_species'
    elif eval_subset == 'cross-family':
        categories = ['all']
        modified_split = split + '_cross_family'
    else:
        raise ValueError(f"Unknown eval_subset: {eval_subset}")
    
    categories = sorted(categories)
    
    # Remove "king cheetah" from validation set as it's not present
    if split == 'val' and 'king cheetah' in categories:
        categories.remove('king cheetah')
    
    return categories, modified_split


class AP10KDataset(torch.utils.data.Dataset):
    """AP-10K Dataset for semantic correspondence evaluation."""

    def __init__(
        self,
        root,
        split,
        image_size=840,
        image_mean="imagenet",
        use_bbox=True,
        class_name=None,
        num_instances=None,
        eval_subset='intra-species',
    ):
        """
        Constructs the AP-10K Dataset loader.

        Args:
            root: Dataset root path
            split: Dataset split ('test', 'val', 'trn')
            image_size: Target image size (square)
            image_mean: Normalization type ('imagenet', 'clip', 'raw', etc.)
            use_bbox: Whether to use bounding boxes for cropping
            class_name: Specific class to evaluate (None for all)
            num_instances: Maximum number of instances to use (None for all)
            eval_subset: Evaluation subset type ('intra-species', 'cross-species', 'cross-family')
        """
        super().__init__()
        
        # Map split names
        split_names = {"train": "trn", "valid": "val", "test": "test"}
        self.split = split_names.get(split, split)
        
        self.root = root
        self.image_size = image_size
        self.use_bbox = use_bbox
        self.eval_subset = eval_subset
        self.class_name = class_name

        # Setup image normalization
        if image_mean == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        elif image_mean == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif image_mean in ["raw", "zeros"]:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        elif image_mean in ["perception", "halves"]:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        elif isinstance(image_mean, (list, tuple)) and len(image_mean) == 3:
            mean = list(image_mean)
            std = [1.0, 1.0, 1.0]
        elif isinstance(image_mean, dict):
            mean = image_mean.get("mean", [0.0, 0.0, 0.0])
            std = image_mean.get("std", [1.0, 1.0, 1.0])
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

        # Get modified split name based on eval_subset
        _, self.modified_split = get_ap10k_categories(root, eval_subset, self.split)
        
        # Load pair annotations
        self.instances = self.get_pair_annotations(class_name)

        if num_instances and len(self.instances) > num_instances:
            random.seed(20)
            random.shuffle(self.instances)
            self.instances = self.instances[:num_instances]

    def get_pair_annotations(self, class_name=None):
        """Load pair annotations from JSON files."""
        if class_name:
            pattern = f'{self.root}/PairAnnotation/{self.modified_split}/*:{class_name}.json'
        else:
            pattern = f'{self.root}/PairAnnotation/{self.modified_split}/*.json'
        
        pair_files = sorted(glob.glob(pattern))
        
        instances = []
        for pair_file in pair_files:
            with open(pair_file) as f:
                data = json.load(f)
            
            # Extract category from filename if not in data
            if 'category' not in data:
                # Filename format: something:category.json
                filename = os.path.basename(pair_file)
                category = filename.split(':')[-1].replace('.json', '')
                data['category'] = category
            
            data['pair_file'] = pair_file
            instances.append(data)
        
        return instances

    def preprocess_kps_pad(self, kps, w, h, size):
        """
        Preprocess keypoints with padding to square.
        
        Args:
            kps: Keypoints tensor (N, 3) with (x, y, visibility)
            w: Original image width
            h: Original image height
            size: Target size
        
        Returns:
            kps: Processed keypoints
            x_offset: X offset after padding
            y_offset: Y offset after padding  
            scale: Scale factor applied
        """
        # Calculate scale to fit image in target size
        scale = size / max(w, h)
        
        # Calculate offsets for centering (padding)
        new_w = w * scale
        new_h = h * scale
        x_offset = (size - new_w) / 2
        y_offset = (size - new_h) / 2
        
        # Scale and offset keypoints
        kps = kps.clone()
        valid = kps[:, 2] > 0
        kps[valid, 0] = kps[valid, 0] * scale + x_offset
        kps[valid, 1] = kps[valid, 1] * scale + y_offset
        
        return kps, x_offset, y_offset, scale

    def load_image_annotation(self, json_path):
        """Load image annotation from JSON file."""
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_image_from_annotation(self, json_path):
        """Get image path from annotation JSON path."""
        img_path = json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
        return img_path

    def __getitem__(self, index):
        """
        Get a pair of images with keypoints.
        
        Returns:
            img_i: Source image tensor
            seg_i: Source mask tensor (placeholder, same as image)
            kps_i: Source keypoints (N, 3)
            img_j: Target image tensor
            seg_j: Target mask tensor (placeholder, same as image)
            kps_j: Target keypoints (N, 3)
            thresh_scale: Threshold scale for PCK computation
            class_name: Category name
        """
        pair_data = self.instances[index]
        
        # Get source and target JSON paths (relative paths in annotation, need to prepend root)
        src_json_path = pair_data["src_json_path"]
        trg_json_path = pair_data["trg_json_path"]
        
        # Prepend data root if paths are not absolute
        if not os.path.isabs(src_json_path):
            src_json_path = src_json_path.replace('data/ap-10k', self.root)
        if not os.path.isabs(trg_json_path):
            trg_json_path = trg_json_path.replace('data/ap-10k', self.root)
        
        # Load annotations
        src_anno = self.load_image_annotation(src_json_path)
        trg_anno = self.load_image_annotation(trg_json_path)
        
        # Get image paths
        src_img_path = self.get_image_from_annotation(src_json_path)
        trg_img_path = self.get_image_from_annotation(trg_json_path)
        
        # Get bounding boxes
        src_bbox = np.asarray(src_anno["bbox"])  # [l, t, w, h]
        trg_bbox = np.asarray(trg_anno["bbox"])
        
        # Get image sizes
        src_size = (src_anno["width"], src_anno["height"])
        trg_size = (trg_anno["width"], trg_anno["height"])
        
        # Load keypoints
        src_kps = torch.tensor(src_anno["keypoints"]).view(-1, 3).float()
        src_kps[:, -1] /= 2  # Convert visibility from COCO format
        
        trg_kps = torch.tensor(trg_anno["keypoints"]).view(-1, 3).float()
        trg_kps[:, -1] /= 2
        
        # Preprocess keypoints with padding
        src_kps, src_x, src_y, src_scale = self.preprocess_kps_pad(
            src_kps, src_size[0], src_size[1], self.image_size
        )
        trg_kps, trg_x, trg_y, trg_scale = self.preprocess_kps_pad(
            trg_kps, trg_size[0], trg_size[1], self.image_size
        )
        
        # Load and process images
        img_i = self.load_and_pad_image(src_img_path)
        img_j = self.load_and_pad_image(trg_img_path)
        
        # Apply transforms
        img_i = self.image_transform(img_i)
        img_j = self.image_transform(img_j)
        
        # Use images as placeholders for masks
        seg_i = img_i
        seg_j = img_j
        
        # Compute threshold scale based on target bounding box
        # bbox format: [l, t, w, h]
        thresh_scale = max(trg_bbox[2], trg_bbox[3]) * trg_scale / self.image_size
        
        # Get class name
        class_name = pair_data.get('category', 'unknown')
        
        return img_i, seg_i, src_kps, img_j, seg_j, trg_kps, thresh_scale, class_name

    def load_and_pad_image(self, img_path):
        """Load image and pad to square."""
        with Image.open(img_path) as f:
            image = np.array(f.convert('RGB'))
        
        h, w = image.shape[:2]
        max_hw = max(h, w)
        
        # Pad to square
        pad_h = (max_hw - h) // 2
        pad_w = (max_hw - w) // 2
        
        image = np.pad(
            image,
            ((pad_h, max_hw - h - pad_h), (pad_w, max_hw - w - pad_w), (0, 0)),
            mode='constant',
            constant_values=255
        )
        
        return Image.fromarray(image)

    def __len__(self):
        return len(self.instances)
