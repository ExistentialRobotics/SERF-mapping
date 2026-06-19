"""Tracking I/O helpers."""

import json
import os

import h5py
import numpy as np

from tracking.config.tracking_config import TrackingConfig


def get_exported_neural_points_task_dir(run_dir: str) -> str:
    """
    Resolve the canonical exported neural point directory for a task run.

    Example:
        /tmp/points/models/task-0021 -> /tmp/points/exported_neural_points/task-0021
    """
    run_dir = run_dir.rstrip("/")
    task_name = os.path.basename(run_dir)
    root_dir = os.path.dirname(os.path.dirname(run_dir))
    return os.path.join(root_dir, "exported_neural_points", task_name)

def write_neural_points_to_hdf5(
    hf: h5py.File,
    points: np.ndarray,
    features: np.ndarray,
    instance_ids: np.ndarray,
    instance_id_map: dict,
    exclude_ids: set[int] | None = None,
) -> None:
    """Write base neural point data to an open HDF5 file.

    Filters out excluded instance IDs before writing (if provided).
    Both ``export_neural_points.py`` and ``neural_point_tracking.py`` call
    this to ensure an identical base format.

    Args:
        hf: Open HDF5 file handle (write mode).
        points: (N, 3) float32 point positions.
        features: (N, D) float32 per-point features.
        instance_ids: (N,) int64 per-point instance IDs.
        instance_id_map: Instance ID to name mapping.
        exclude_ids: Optional set of instance IDs to filter out.
    """
    if exclude_ids:
        mask = ~np.isin(instance_ids, list(exclude_ids))
        points = points[mask]
        features = features[mask]
        instance_ids = instance_ids[mask]

    unique_ids = sorted(set(instance_ids.tolist()))

    hf.create_dataset("initial_points", data=points)
    hf.create_dataset("initial_features", data=features)
    hf.create_dataset("initial_instance_ids", data=instance_ids)
    hf.attrs["num_instances"] = len(unique_ids)
    hf.attrs["instance_ids"] = json.dumps(unique_ids)
    hf.attrs["instance_id_to_name"] = json.dumps(
        {str(k): v for k, v in instance_id_map.items()}
    )

def export_tracking_results(
    cfg: TrackingConfig,
    episode_name: str,
    initial_points: np.ndarray,
    initial_features: np.ndarray,
    initial_instance_ids: np.ndarray,
    frame_indices: list[int],
    frame_transforms: dict[int, list[np.ndarray]],
    training_id_map: dict[str, str],
    exclude_ids: set[int],
) -> None:
    """Save tracking results (transforms, initial state) to HDF5."""
    output_dir = os.path.join(get_exported_neural_points_task_dir(cfg.run_dir), "train")
    os.makedirs(output_dir, exist_ok=True)
    output_hdf5_path = os.path.join(output_dir, f"{episode_name}.hdf5")

    with h5py.File(output_hdf5_path, "w") as hf:
        # Base data (identical to export_neural_points.py)
        write_neural_points_to_hdf5(
            hf, initial_points, initial_features, initial_instance_ids,
            training_id_map, exclude_ids=exclude_ids,
        )
        # Tracking-specific data
        hf.create_dataset("frame_indices", data=np.array(frame_indices, dtype=np.int64))
        grp = hf.create_group("transforms")
        for iid, tf_list in frame_transforms.items():
            grp.create_dataset(str(iid), data=np.stack(tf_list).astype(np.float32))
        hf.attrs["num_frames"] = len(frame_indices)

    print(f"[DONE] Tracking results saved to {output_hdf5_path}")
