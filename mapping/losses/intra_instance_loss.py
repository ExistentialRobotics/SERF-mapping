"""
Intra-instance contrastive loss for part-aware feature learning.

Same SAM segment within same instance = positive pairs (pull together)
Different SAM segment within same instance = negative pairs (push apart)

Only considers pairs within the same instance (intra-instance only).
Uses InfoNCE/SupCon style loss.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .inter_category_loss import (
    CategoryMapper,
    compute_importance_weights,
    importance_sample,
    compute_infonce_loss,
)


def intra_instance_loss(
    features: torch.Tensor,
    sam_mask_ids: torch.Tensor,
    instance_ids: Optional[torch.Tensor] = None,
    inst_to_category: Optional[Dict[int, str]] = None,
    temperature: float = 0.1,
    max_samples: int = 256,
    mapper: Optional[CategoryMapper] = None,
) -> torch.Tensor:
    """
    Supervised contrastive loss using SAM segments (intra-instance only).

    Same SAM segment within same instance = positive pairs.
    Different SAM segments within same instance = negative pairs.
    Only considers pairs within the same instance.
    Uses InfoNCE/SupCon style for better representation learning.

    Uses per-category normalized importance sampling: each category contributes
    equally regardless of point count.

    Args:
        features: (N, D) latent features for points
        sam_mask_ids: (N,) SAM mask IDs for each point (-1 for background)
        instance_ids: (N,) instance IDs (optional)
        inst_to_category: mapping from instance_id to category name for importance sampling (optional)
        temperature: Softmax temperature (lower = sharper distribution)
        max_samples: Max samples for memory efficiency
        mapper: pre-built CategoryMapper (avoids re-creation each step)

    Returns:
        Scalar contrastive loss
    """
    device = features.device
    N = features.shape[0]

    if N < 2:
        return torch.tensor(0.0, device=device)

    # Filter valid samples (mask_id >= 0)
    valid_mask = sam_mask_ids >= 0
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=device)

    features_valid = features[valid_mask]
    mask_ids_valid = sam_mask_ids[valid_mask]
    inst_ids_valid = instance_ids[valid_mask] if instance_ids is not None else None
    M = features_valid.shape[0]

    # Importance sampling
    num_samples = min(M, max_samples)
    if inst_ids_valid is not None and inst_to_category:
        if mapper is None:
            mapper = CategoryMapper(inst_to_category, device=device)
        cat_ids_valid = mapper.get_category_ids(inst_ids_valid)
        weights = compute_importance_weights(cat_ids_valid)
        [features_valid, mask_ids_valid, inst_ids_valid] = importance_sample(
            [features_valid, mask_ids_valid, inst_ids_valid], weights, num_samples
        )
    else:
        # Uniform sampling if no category info
        perm = torch.randperm(M, device=device)[:num_samples]
        features_valid = features_valid[perm]
        mask_ids_valid = mask_ids_valid[perm]
        if inst_ids_valid is not None:
            inst_ids_valid = inst_ids_valid[perm]
    M = num_samples

    # Normalize features
    features_valid = F.normalize(features_valid, dim=-1)

    # Same-segment mask
    same_segment = mask_ids_valid.unsqueeze(0) == mask_ids_valid.unsqueeze(1)
    same_segment.fill_diagonal_(False)

    # Same-instance mask (for restricting negatives to same instance)
    if inst_ids_valid is not None:
        same_instance = inst_ids_valid.unsqueeze(0) == inst_ids_valid.unsqueeze(1)
    else:
        same_instance = torch.ones(M, M, dtype=torch.bool, device=device)

    # Positive: same instance AND same segment
    positive_mask = same_segment & same_instance

    # Valid pairs for softmax: same instance only (exclude self)
    # Only intra-instance pairs are considered (no cross-instance negatives)
    valid_mask = same_instance.clone()
    valid_mask.fill_diagonal_(False)

    return compute_infonce_loss(features_valid, positive_mask, valid_mask, temperature)
