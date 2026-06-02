
from typing import Sequence

import torch
import torch.nn as nn


def _flatten_feat(feat: torch.Tensor) -> torch.Tensor:
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3))
    if feat.dim() == 2:
        return feat
    raise ValueError(f"Unsupported feature shape {feat.shape}")


class PoseLinearProbe(nn.Module):
    """
    Minimal linear probe that consumes CLS or globally pooled dense tokens
    from frozen backbones and predicts discretized azimuth/elevation/theta bins.
    """

    def __init__(
        self,
        feat_dims: Sequence[int],
        num_bins: int = 40,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(feat_dims, int):
            feat_dims = [feat_dims]
        self.feat_dims = list(feat_dims)
        in_dim = sum(self.feat_dims)

        if hidden_dim is not None and hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_dim = hidden_dim
        else:
            self.proj = None
            head_dim = in_dim

        self.azimuth_head = nn.Linear(head_dim, num_bins)
        self.elevation_head = nn.Linear(head_dim, num_bins)
        self.theta_head = nn.Linear(head_dim, num_bins)

    def forward(self, feats):
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        pooled = [_flatten_feat(f) for f in feats]
        concat = torch.cat(pooled, dim=-1)
        if self.proj is not None:
            concat = self.proj(concat)
        az = self.azimuth_head(concat)
        el = self.elevation_head(concat)
        th = self.theta_head(concat)
        return az, el, th
