import os
import random

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def set_seed(seed: int):
    """Seed Python, NumPy, Torch, and CUDA RNGs for reproducible probe training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ddp_setup(rank: int, world_size: int, port: int = 12355):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup(world_size: int):
    """Destroy DDP process group if active."""
    if world_size > 1:
        destroy_process_group()


def unwrap_model(model):
    """Get the underlying model whether DDP-wrapped or not."""
    return model.module if isinstance(model, DDP) else model
