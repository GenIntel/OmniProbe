
import io
import os
import pickle
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _decode_frame(frame_bytes: bytes) -> np.ndarray:
    """Decode TAP-Vid JPEG bytes into an RGB array."""
    buffer = io.BytesIO(frame_bytes)
    image = Image.open(buffer)
    return np.array(image)


def _resize_video(frames: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a numpy video clip to (size[0], size[1]) using bilinear filtering."""
    if frames.shape[1:3] == size:
        return frames

    tensor = (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)
        .float()
    )  # (T, 3, H, W)
    tensor = torch.nn.functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )
    tensor = tensor.permute(0, 2, 3, 1).byte().numpy()
    return tensor


class TAPVidDataset(Dataset):
    """
    Hydra-compatible wrapper around the TAP-Vid evaluation splits used for
    zero-shot point tracking.
    """

    def __init__(
        self,
        tapvid_root: str,
        dataset_type: str = "davis_first",
        resize_to: int = 256,
        query_stride: int = 5,
        max_videos: int | None = None,
        **_: dict,
    ) -> None:
        super().__init__()
        self.tapvid_root = tapvid_root
        self.dataset_type = dataset_type
        self.resize_to = resize_to
        self.query_mode = "first" if "first" in dataset_type else "strided"
        self.query_stride = query_stride
        self.name = f"tapvid_{dataset_type}"

        if "kinetics" in dataset_type:
            pickle_paths = glob(os.path.join(tapvid_root, "*_of_0010.pkl"))
            points_dataset: List[Dict] = []
            for path in pickle_paths:
                with open(path, "rb") as handle:
                    points_dataset.extend(pickle.load(handle))
            self.points_dataset = points_dataset
            self.video_keys = list(range(len(self.points_dataset)))
        else:
            with open(tapvid_root, "rb") as handle:
                self.points_dataset = pickle.load(handle)
            if isinstance(self.points_dataset, dict):
                self.video_keys = sorted(self.points_dataset.keys())
            else:
                self.video_keys = list(range(len(self.points_dataset)))

        if max_videos is not None:
            self.video_keys = self.video_keys[:max_videos]

        if len(self.video_keys) == 0:
            raise RuntimeError(
                f"No TAP-Vid videos found under {tapvid_root} ({dataset_type})."
            )

    def __len__(self) -> int:
        return len(self.video_keys)

    def _fetch_entry(self, index: int) -> Dict:
        key = self.video_keys[index]
        if isinstance(self.points_dataset, dict):
            return self.points_dataset[key]
        return self.points_dataset[key]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self._fetch_entry(index)
        frames = entry["video"].copy()

        if isinstance(frames[0], bytes):
            frames = np.array([_decode_frame(frame) for frame in frames])

        if self.resize_to is not None:
            frames = _resize_video(frames, (self.resize_to, self.resize_to))
            scale = np.array([self.resize_to, self.resize_to], dtype=np.float32)
        else:
            scale = np.array([frames.shape[2], frames.shape[1]], dtype=np.float32)

        target_points = entry["points"].copy() * scale
        target_occ = entry["occluded"].copy()

        if self.query_mode == "first":
            valid = np.sum(~target_occ, axis=1) > 0
            target_points = target_points[valid]
            target_occ = target_occ[valid]
            query_points = []
            for track_occ, track_pts in zip(target_occ, target_points):
                first_visible = np.where(track_occ == 0)[0][0]
                x = track_pts[first_visible, 0]
                y = track_pts[first_visible, 1]
                query_points.append(np.array([first_visible, y, x]))
            query_points = np.stack(query_points, axis=0)
        else:
            stride = self.query_stride
            queries = []
            tracks = []
            occs = []
            total = target_occ.shape[0]
            for frame_idx in range(0, target_occ.shape[1], stride):
                visible_mask = target_occ[:, frame_idx] == 0
                if not visible_mask.any():
                    continue
                q = np.stack(
                    [
                        frame_idx * np.ones(total),
                        target_points[:, frame_idx, 1],
                        target_points[:, frame_idx, 0],
                    ],
                    axis=-1,
                )
                queries.append(q[visible_mask])
                tracks.append(target_points[visible_mask])
                occs.append(target_occ[visible_mask])
            query_points = np.concatenate(queries, axis=0)
            target_points = np.concatenate(tracks, axis=0)
            target_occ = np.concatenate(occs, axis=0)

        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        trajectories = torch.from_numpy(target_points).permute(1, 0, 2).float()
        visibility = torch.logical_not(torch.from_numpy(target_occ)).permute(1, 0)
        query_points_tensor = torch.from_numpy(query_points).float()

        return {
            "video": video,
            "trajectory": trajectories,
            "visibility": visibility,
            "query_points": query_points_tensor,
            "video_name": self.video_keys[index]
        }