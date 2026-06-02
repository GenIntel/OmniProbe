from collections.abc import Sequence

import torch.nn as nn

from .ep import EfficientProbing


class PoseEPProbe(nn.Module):
    def __init__(
        self,
        feat_dims: Sequence[int],
        num_bins: int = 40,
        num_queries: int = 8,
        num_heads: int = 1,
        **_,
    ):
        super().__init__()
        if isinstance(feat_dims, int):
            feat_dims = [feat_dims]
        in_dim = feat_dims[-1]
        self.ep = EfficientProbing(dim=in_dim, num_queries=num_queries, num_heads=num_heads)
        self.azimuth_head = nn.Linear(in_dim, num_bins)
        self.elevation_head = nn.Linear(in_dim, num_bins)
        self.theta_head = nn.Linear(in_dim, num_bins)

    def forward(self, feats):
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.dim() == 4:
            batch_size, channels, height, width = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        pooled = self.ep(feats)
        return (
            self.azimuth_head(pooled),
            self.elevation_head(pooled),
            self.theta_head(pooled),
        )
