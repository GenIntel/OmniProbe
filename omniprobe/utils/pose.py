
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.linalg import logm


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def continuous_to_bin(
    values,
    *,
    num_bins: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
    border_type: str = "periodic",
):
    assert border_type in {"periodic", "clamp"}
    arr = _to_numpy(values)
    scale = (arr - min_value) / (max_value - min_value)
    if border_type == "periodic":
        scale = np.mod(scale, 1.0)
    else:
        scale = np.clip(scale, 0.0, 1.0)
    bins = np.floor(scale * num_bins).astype(np.int64) % num_bins
    if isinstance(values, torch.Tensor):
        return torch.as_tensor(bins, dtype=torch.long, device=values.device)
    if np.isscalar(values):
        return int(bins)
    return bins


def bin_to_continuous(
    bins,
    *,
    num_bins: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    arr = _to_numpy(bins)
    arr = np.mod(arr, num_bins)
    ratio = (arr + 0.5) / num_bins
    values = ratio * (max_value - min_value) + min_value
    if isinstance(bins, torch.Tensor):
        return torch.as_tensor(values, dtype=torch.float32, device=bins.device)
    return values


def _rotation_from_angles(theta, elevation, azimuth):
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)
    rz = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0],
            [np.sin(azimuth), np.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(elevation), -np.sin(elevation)],
            [0, np.sin(elevation), np.cos(elevation)],
        ]
    )
    ry = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return ry @ (rx @ rz)


def pose_error(pose_a, pose_b) -> float:
    mat_a = _rotation_from_angles(*pose_a)
    mat_b = _rotation_from_angles(*pose_b)
    delta = logm(mat_b.T @ mat_a)
    err = np.sqrt((delta**2).sum()) / np.sqrt(2)
    return float(np.real(err))


def batch_pose_error(pose_a, pose_b):
    az_a, el_a, th_a = pose_a
    az_b, el_b, th_b = pose_b
    az_a = np.asarray(az_a)
    el_a = np.asarray(el_a)
    th_a = np.asarray(th_a)
    az_b = np.asarray(az_b)
    el_b = np.asarray(el_b)
    th_b = np.asarray(th_b)

    mats_a = np.stack(
        [
            _rotation_from_angles(t, e, a)
            for t, e, a in zip(th_a.flatten(), el_a.flatten(), az_a.flatten())
        ]
    )
    mats_b = np.stack(
        [
            _rotation_from_angles(t, e, a)
            for t, e, a in zip(th_b.flatten(), el_b.flatten(), az_b.flatten())
        ]
    )
    delta = np.einsum("bij,bjk->bik", np.transpose(mats_b, (0, 2, 1)), mats_a)
    logs = np.array([logm(m) for m in delta])
    errors = np.sqrt((logs.real ** 2).sum(axis=(1, 2)) / 2.0)
    return errors
