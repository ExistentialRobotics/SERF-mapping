"""
Category-aware contrastive loss for semantic feature learning.

Same object CATEGORY = positive pairs, different category = negative pairs.

Also provides utility functions:
- CategoryMapper: maps instance IDs to category IDs
- compute_importance_weights: per-category normalized sampling weights
- importance_sample: samples tensors with importance weights
- compute_infonce_loss: computes InfoNCE/SupCon style contrastive loss
- inter_category_loss: single-scene InfoNCE loss on interpolated features (primary)
- compute_cross_scene_inter_loss: cross-scene balanced loss with interpolated features
  (one call per scene via interpolate_features; balanced n_per_category across all scenes;
   robot or any extra category supported via extra_features)
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Set

# Only exclude categories that are not mapped (background, unknown)
DEFAULT_EXCLUDE_CATEGORIES = {'background', 'unknown'}


class CategoryMapper:
    """
    Cached mapper from instance_id to category_id.
    Avoids Python loops during training.
    """

    def __init__(
        self,
        inst_to_category: Dict[int, str],
        exclude_categories: Optional[Set[str]] = None,
        device: str = "cuda",
    ):
        if exclude_categories is None:
            exclude_categories = DEFAULT_EXCLUDE_CATEGORIES

        # Build category list
        unique_categories = sorted(set(inst_to_category.values()) - exclude_categories)
        self.cat_to_id = {cat: i for i, cat in enumerate(unique_categories)}
        self.num_categories = len(unique_categories)

        # Create lookup table: inst_id -> category_id (-1 for excluded)
        max_inst_id = max((int(k) for k in inst_to_category.keys()), default=0)

        lookup = torch.full((max_inst_id + 1,), -1, dtype=torch.long, device=device)
        for inst_id, cat in inst_to_category.items():
            inst_id = int(inst_id)
            if cat in self.cat_to_id:
                lookup[inst_id] = self.cat_to_id[cat]

        self.lookup = lookup
        self.device = device

    def get_category_ids(self, instance_ids: torch.Tensor) -> torch.Tensor:
        """Convert instance_ids to category_ids using lookup table."""
        safe_ids = instance_ids.clamp(0, len(self.lookup) - 1)
        return self.lookup[safe_ids]


def compute_importance_weights(category_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute per-category normalized importance sampling weights.

    Each category contributes equally to the sampling distribution,
    regardless of the number of points in that category.

    Returns:
        Normalized probability weights for sampling.
    """
    _, inverse, counts = torch.unique(category_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].float()
    return weights / weights.sum()


def importance_sample(
    tensors: list,
    weights: torch.Tensor,
    num_samples: int,
) -> list:
    """
    Apply importance sampling to multiple tensors.

    Args:
        tensors: List of tensors to sample from (same first dimension)
        weights: Normalized sampling weights
        num_samples: Number of samples to draw

    Returns:
        List of sampled tensors
    """
    sampled_indices = torch.multinomial(weights, num_samples, replacement=False)
    return [t[sampled_indices] if t is not None else None for t in tensors]


def compute_infonce_loss(
    features: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Compute InfoNCE/SupCon style contrastive loss.

    Args:
        features: (M, D) normalized features
        positive_mask: (M, M) boolean mask for positive pairs
        valid_mask: (M, M) boolean mask for valid pairs (included in softmax)
        temperature: Softmax temperature

    Returns:
        Scalar contrastive loss
    """
    device = features.device
    M = features.shape[0]

    # Similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature

    # Mask out invalid pairs from softmax
    sim_matrix = sim_matrix.masked_fill(~valid_mask, float('-inf'))
    log_softmax = F.log_softmax(sim_matrix, dim=1)
    log_softmax = log_softmax.masked_fill(~valid_mask, 0.0)

    # Count positives per anchor
    pos_counts = positive_mask.sum(dim=1)
    has_positives = pos_counts > 0

    if has_positives.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Compute loss for anchors with positives
    pos_log_sum = (log_softmax[has_positives] * positive_mask[has_positives].float()).sum(dim=1)
    loss = -(pos_log_sum / pos_counts[has_positives].float()).mean()

    return loss


def compute_cross_scene_inter_loss(
    point_maps: Dict[str, Any],
    mapper: "CategoryMapper",
    max_total_samples: int,
    temperature: float,
    device: str,
    extra_features: Optional[torch.Tensor] = None,
    extra_category_id: int = -1,
) -> torch.Tensor:
    """
    Cross-scene inter-category contrastive loss using interpolated features.

    Collects interpolated features from all NeuralPointMaps across all scenes,
    then samples equally per category so that each contributes the same number
    of features. Total features ≤ max_total_samples.

    Robot (or any extra category) is included via extra_features and counts
    as one additional category, receiving the same n_per_category samples.

    Args:
        point_maps: Dict[str, NeuralPointMap] — all env scene point maps.
        mapper: Shared CategoryMapper for instance_id → category_id mapping.
        max_total_samples: Total feature budget; divided equally among all
            present categories: n_per_category = max_total_samples // K.
        temperature: InfoNCE softmax temperature.
        device: Torch device string.
        extra_features: Optional (M, D) pre-computed interpolated features for
            an extra category (e.g., robot surface features).
        extra_category_id: Category ID for extra_features.

    Returns:
        Scalar contrastive loss, 0.0 if fewer than 2 categories present.
    """
    feat_bank: Dict[int, List[torch.Tensor]] = {}

    for pm in point_maps.values():
        if pm.map_points.shape[0] == 0:
            continue

        inst_ids = pm.map_instance_ids          # (N,)
        cat_ids = mapper.get_category_ids(inst_ids)  # (N,)
        valid = cat_ids >= 0
        if valid.sum() < 2:
            continue

        valid_points = pm.map_points[valid]     # (N_valid, 3)
        valid_cat_ids = cat_ids[valid]          # (N_valid,)

        # Cap per-scene to avoid excessive interpolation queries
        N = valid_points.shape[0]
        if N > max_total_samples:
            perm = torch.randperm(N, device=device)[:max_total_samples]
            valid_points = valid_points[perm]
            valid_cat_ids = valid_cat_ids[perm]

        # Single interpolate_features call per scene (efficient batch)
        interp_feats = pm.interpolate_features(valid_points)  # (N_capped, D)

        for cat_id_val in valid_cat_ids.unique():
            cid = cat_id_val.item()
            mask = valid_cat_ids == cat_id_val
            if cid not in feat_bank:
                feat_bank[cid] = []
            feat_bank[cid].append(interp_feats[mask])

    # --- Extra category (e.g., robot, pre-computed interpolated features) ---
    if extra_features is not None and extra_category_id >= 0 and extra_features.shape[0] > 0:
        feat_bank[extra_category_id] = [extra_features]

    K = len(feat_bank)
    if K < 2:
        return torch.tensor(0.0, device=device)

    # n_per_category: divide total budget equally among present categories
    n_per_category = max(1, max_total_samples // K)

    all_feats: List[torch.Tensor] = []
    all_cat_ids: List[torch.Tensor] = []

    for cid, feat_list in feat_bank.items():
        cat_feats = torch.cat(feat_list, dim=0)          # (N_cat_total, D)
        n = min(n_per_category, cat_feats.shape[0])
        idx = torch.randperm(cat_feats.shape[0], device=device)[:n]
        all_feats.append(cat_feats[idx])
        all_cat_ids.append(torch.full((n,), cid, dtype=torch.long, device=device))

    if len(all_feats) < 2:
        return torch.tensor(0.0, device=device)

    combined_feats = torch.cat(all_feats, dim=0)        # (M, D)
    combined_cat_ids = torch.cat(all_cat_ids, dim=0)    # (M,)

    if combined_cat_ids.unique().numel() < 2:
        return torch.tensor(0.0, device=device)

    M = combined_feats.shape[0]
    normalized = F.normalize(combined_feats, dim=-1)
    positive_mask = combined_cat_ids.unsqueeze(0) == combined_cat_ids.unsqueeze(1)
    positive_mask.fill_diagonal_(False)
    valid_mask = ~torch.eye(M, dtype=torch.bool, device=device)

    return compute_infonce_loss(normalized, positive_mask, valid_mask, temperature)


def inter_category_loss(
    features: torch.Tensor,
    instance_ids: torch.Tensor,
    inst_to_category: Dict[int, str],
    temperature: float = 0.1,
    max_samples: int = 256,
    mapper: Optional[CategoryMapper] = None,
) -> torch.Tensor:
    """
    Supervised contrastive loss: same category = positive, different = negative.

    Uses per-category normalized importance sampling: each category contributes
    equally regardless of point count.

    Args:
        features: (N, D) predicted features
        instance_ids: (N,) instance IDs
        inst_to_category: mapping from instance_id to category name
        temperature: softmax temperature
        max_samples: max samples for memory efficiency
        mapper: pre-built CategoryMapper (avoids re-creation each step)

    Returns:
        Scalar contrastive loss
    """
    device = features.device

    # Use provided mapper or create a new one
    if mapper is None:
        if not inst_to_category:
            return torch.tensor(0.0, device=device)
        mapper = CategoryMapper(inst_to_category, device=device)
    if mapper.num_categories < 2:
        return torch.tensor(0.0, device=device)

    category_ids = mapper.get_category_ids(instance_ids)

    # Filter valid samples
    valid_mask = category_ids >= 0
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=device)

    features_valid = features[valid_mask]
    category_ids_valid = category_ids[valid_mask]
    M = features_valid.shape[0]

    # Importance sampling
    weights = compute_importance_weights(category_ids_valid)
    num_samples = min(M, max_samples)
    [features_valid, category_ids_valid] = importance_sample(
        [features_valid, category_ids_valid], weights, num_samples
    )
    M = num_samples

    # Normalize features
    features_valid = F.normalize(features_valid, dim=-1)

    positive_mask = category_ids_valid.unsqueeze(0) == category_ids_valid.unsqueeze(1)
    positive_mask.fill_diagonal_(False)
    valid_mask = ~torch.eye(M, dtype=torch.bool, device=device)

    return compute_infonce_loss(features_valid, positive_mask, valid_mask, temperature)
