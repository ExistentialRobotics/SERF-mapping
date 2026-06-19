"""Fixed-topology neural point representation for robot surfaces.

Unlike NeuralPointMap which dynamically registers points with a voxel hash,
RobotNeuralPointMap has a FIXED set of N surface points (from RobotSurfaceSampler)
with learnable features. Point positions change via FK, features are
configuration-invariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobotNeuralPointMap(nn.Module):
    """Fixed-topology neural point map for a robot.

    Attributes:
        features: (N, feature_dim) nn.Parameter — learnable per-point features.
        knn_k: Number of nearest neighbors for interpolation.
        temperature: Softmax temperature for distance weighting.
    """

    def __init__(
        self,
        n_points: int,
        feature_dim: int = 64,
        knn_k: int = 10,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.knn_k = knn_k
        self.temperature = temperature
        self.features = nn.Parameter(torch.zeros(n_points, feature_dim))
        nn.init.normal_(self.features, mean=0.0, std=0.01)

    def interpolate_features(
        self,
        query_points: torch.Tensor,
        surface_points: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate learned features at query points via KNN on FK-posed surface.

        Args:
            query_points: (Q, 3) observed 3D world points.
            surface_points: (N, 3) FK-transformed surface points (current config).

        Returns:
            (Q, feature_dim) interpolated features.
        """
        dists = torch.cdist(query_points, surface_points)  # (Q, N)
        topk_dists, topk_idx = dists.topk(self.knn_k, dim=-1, largest=False)

        weights = F.softmax(-topk_dists / self.temperature, dim=-1)  # (Q, K)
        neighbor_feats = self.features[topk_idx]  # (Q, K, D)
        return (weights.unsqueeze(-1) * neighbor_feats).sum(dim=1)  # (Q, D)
