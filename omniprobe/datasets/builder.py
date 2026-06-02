import os
from typing import Optional

import torch
from hydra.utils import instantiate
from PIL import ImageFile
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# avoid open file error
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")


def build_loader(
    cfg,
    split,
    batch_size,
    num_gpus=1,
    num_workers: Optional[int] = None,
    **kwargs,
):
    """
    Build a PyTorch dataloader and the underlying dataset (using config).
    """
    # Build a dataset from the provided dataset config.
    dataset = instantiate(cfg, split=split, **kwargs)

    use_ddp = num_gpus > 1
    sampler = DistributedSampler(dataset) if use_ddp else None
    shuffle = (split == "train") and not use_ddp
    if num_workers is None:
        n_workers = min(len(os.sched_getaffinity(0)), 2)
    else:
        n_workers = num_workers

    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=shuffle,
        sampler=sampler,
    )

    return loader