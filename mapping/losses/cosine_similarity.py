"""
Cosine similarity reconstruction loss.
"""

import torch
import torch.nn.functional as F


def cosine_similarity_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute reconstruction loss using cosine similarity.

    Args:
        pred_feat: (N, D) predicted features
        target_feat: (N, D) target features (e.g., DINOv2)

    Returns:
        loss: Scalar reconstruction loss (1 - mean cosine similarity)
        cos_sim: (N,) cosine similarity per sample
    """
    cos_sim = F.cosine_similarity(pred_feat, target_feat, dim=-1)
    loss = 1.0 - cos_sim.mean()
    return loss, cos_sim
