"""Point map loading and saving utilities."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from mapping.config.env_train_config import EnvTrainConfig
from mapping.representations.neural_point_map import NeuralPointMap


def find_neural_point_file(run_dir: str, episode_name: str) -> str:
    """Find a neural point map .pt file for a given episode.

    Searches in order:
    1. ``{run_dir}/neural_points/{episode_name}.pt``
    2. ``{run_dir}/neural_points/train/{episode_name}.pt``

    Args:
        run_dir: Training run directory containing ``neural_points/``.
        episode_name: Episode name (without .pt extension).

    Returns:
        Absolute path to the .pt file.

    Raises:
        FileNotFoundError: If no .pt file is found.
    """
    neural_points_dir = os.path.join(run_dir, "neural_points")
    candidates = [
        os.path.join(neural_points_dir, f"{episode_name}.pt"),
        os.path.join(neural_points_dir, "train", f"{episode_name}.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No exact neural point map found for episode "
        f"'{episode_name}'. Checked: {candidates}"
    )


def _extract_episode_name(env_name: str) -> str:
    """Extract episode/scene name from env_name (e.g. 'cache_sunghwan_episode_00210060' -> 'episode_00210060')."""
    for prefix in ("episode_", "scene_"):
        idx = env_name.find(prefix)
        if idx != -1:
            return env_name[idx:]
    return env_name


def _resolve_split(env_name: str, episode_splits: Optional[Dict[str, str]],
                   default_split: str) -> str:
    """Resolve the split for an env_name using the episode_splits mapping.

    Matches by checking if any key in episode_splits contains the same
    episode/scene name substring as env_name.
    """
    if not episode_splits:
        return default_split
    episode_name = _extract_episode_name(env_name)
    for filename, split in episode_splits.items():
        if episode_name in filename:
            return split
    return default_split


def load_point_map_from_state_dict(
    state_dict: dict,
    device: str = "cpu",
    voxel_size: float = 0.01,
    knn_k: int = 10,
    num_nei_cells: Optional[int] = None,
    search_alpha: float = 1.0,
) -> NeuralPointMap:
    """
    Create and load a NeuralPointMap from a saved state dict.

    Infers dimensions (feature_dim, hash_table_size, num_nei_cells) from the
    state dict, resizes buffers to match, then loads the weights.

    Args:
        state_dict: Saved state dict from a NeuralPointMap checkpoint.
        device: Torch device to place the map on.
        voxel_size: Voxel size for the hash table.
        knn_k: Number of nearest neighbors for queries.
        num_nei_cells: Number of neighbor cells. If None, inferred from state_dict.
        search_alpha: Search radius multiplier.

    Returns:
        Loaded NeuralPointMap on the specified device.
    """
    feature_dim = state_dict["map_features"].shape[1]
    hash_table_size = state_dict["buffer_pt_index"].shape[0]

    # Infer num_nei_cells from saved neighbor_dx if not provided
    if num_nei_cells is None:
        if "neighbor_dx" in state_dict:
            num_nei_cells = int(state_dict["neighbor_dx"].abs().max().item())
        else:
            num_nei_cells = 1

    point_map = NeuralPointMap(
        voxel_size=voxel_size,
        feature_dim=feature_dim,
        knn_k=knn_k,
        hash_table_size=hash_table_size,
        num_nei_cells=num_nei_cells,
        search_alpha=search_alpha,
    )

    # Resize buffers to match saved state
    point_map.register_buffer("map_points", torch.empty_like(state_dict["map_points"]))
    point_map.map_features = torch.nn.Parameter(
        torch.empty_like(state_dict["map_features"])
    )
    if "map_instance_ids" in state_dict:
        point_map.register_buffer(
            "map_instance_ids", torch.empty_like(state_dict["map_instance_ids"])
        )
    if "neighbor_dx" in state_dict:
        point_map._buffers["neighbor_dx"] = torch.empty_like(state_dict["neighbor_dx"])

    point_map.load_state_dict(state_dict)
    return point_map.to(device)


def load_or_create_point_maps(
    env_names: List[str],
    cfg: EnvTrainConfig,
    output_dir: Path,
    device: str,
    split: str = "train",
    episode_splits: Optional[Dict[str, str]] = None,
    require_existing: bool = False,
) -> Dict[str, NeuralPointMap]:
    """
    Load existing point maps or create new ones for each environment.

    Args:
        env_names: List of environment names from dataset
        cfg: Training configuration
        output_dir: Directory containing existing neural_points/
        device: torch device
        split: Default data split ("train" or "eval"), used when episode_splits is empty
        episode_splits: Optional {filename: split} mapping for per-episode split resolution

    Returns:
        point_maps: dict of env_name -> NeuralPointMap (on device)
    """
    point_maps = {}

    for env_name in env_names:
        # Extract episode/scene name from env_name (e.g. "train_episode_00210060")
        episode_name = _extract_episode_name(env_name)
        ep_split = _resolve_split(env_name, episode_splits, split)
        existing_path = output_dir / "neural_points" / ep_split / f"{episode_name}.pt"

        point_map = _try_load_point_map(existing_path, cfg, device)

        if point_map is None:
            if require_existing:
                raise FileNotFoundError(
                    f"Stage 2 requires existing neural points but not found: {existing_path}"
                )
            print(f"\n[INIT] Creating new NeuralPointMap for {env_name}")
            point_map = NeuralPointMap(
                voxel_size=cfg.voxel_size,
                feature_dim=cfg.feature_dim,
                knn_k=cfg.knn_k,
                hash_table_size=cfg.hash_table_size,
                num_nei_cells=cfg.num_nei_cells,
                search_alpha=cfg.search_alpha,
            ).to(device)

        point_maps[env_name] = point_map

    return point_maps


def _try_load_point_map(
    path: Path, cfg: EnvTrainConfig, device: str
) -> Optional[NeuralPointMap]:
    """Try to load a point map from path. Returns None if not found."""
    if not path.exists():
        return None

    print(f"\n[LOAD] Loading existing neural points from {path}")
    try:
        saved_state = torch.load(path, map_location='cpu', weights_only=True)
    except RuntimeError as e:
        print(f"[WARN] Corrupted neural point file {path}: {e}")
        print(f"[WARN] Deleting corrupted file and recreating from scratch.")
        os.remove(path)
        return None

    point_map = load_point_map_from_state_dict(
        saved_state,
        device=device,
        voxel_size=cfg.voxel_size,
        knn_k=cfg.knn_k,
        num_nei_cells=cfg.num_nei_cells,
        search_alpha=cfg.search_alpha,
    )

    active_count = (
        (point_map.buffer_pt_index != point_map.EMPTY_KEY)
        & (point_map.buffer_pt_index != point_map.DELETED_KEY)
    ).sum().item()
    print(
        f"[LOAD] Loaded {saved_state['map_points'].shape[0]} points, "
        f"{active_count} active hash entries"
    )

    return point_map


def save_point_maps(point_maps: Dict[str, NeuralPointMap], output_dir: Path,
                     split: str = "train",
                     episode_splits: Optional[Dict[str, str]] = None):
    """Save all point maps to output_dir/neural_points/{split}/.

    Args:
        episode_splits: Optional {filename: split} mapping for per-episode split resolution.
            If None, uses the single ``split`` parameter for all episodes.
    """
    for env_name, point_map in point_maps.items():
        episode_name = _extract_episode_name(env_name)
        ep_split = _resolve_split(env_name, episode_splits, split)

        neural_points_dir = output_dir / "neural_points" / ep_split
        neural_points_dir.mkdir(parents=True, exist_ok=True)

        save_path = neural_points_dir / f"{episode_name}.pt"
        torch.save(point_map.state_dict(), save_path)
        print(f"[SAVE] Saved {episode_name}.pt ({ep_split})")
