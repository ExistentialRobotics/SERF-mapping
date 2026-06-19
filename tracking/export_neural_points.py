"""Export neural point maps (.pt) to HDF5 format for VLA model input."""
# {episode_name}.hdf5
# ├── initial_points        (N, 3) float32
# ├── initial_features      (N, D) float32
# ├── initial_instance_ids  (N,) int64
# ├── attrs:
# │   ├── num_instances     int
# │   ├── instance_ids      JSON [int, ...]
# │   └── instance_id_to_name  JSON {str: str}

import argparse
import os
import sys

sys.path[0] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import h5py
import numpy as np
import torch
import yaml

from mapping.utils.point_map_io import load_point_map_from_state_dict
from tracking.utils import get_exported_neural_points_task_dir, write_neural_points_to_hdf5
from utils.instance_ids import filter_ids_by_keywords, load_instance_id_map_from_json


def export_single(
    pt_path: str,
    output_path: str,
    model_config: dict,
    instance_id_map: dict,
    exclude_ids: set[int] | None = None,
    device: str = "cpu",
) -> None:
    """Load a neural point map and export initial state to HDF5."""
    state_dict = torch.load(pt_path, map_location="cpu")

    pm_cfg = model_config["point_map"]
    neural_points = load_point_map_from_state_dict(
        state_dict,
        device=device,
        voxel_size=pm_cfg["voxel_size"],
        knn_k=pm_cfg["knn_k"],
        num_nei_cells=pm_cfg.get("num_nei_cells"),
        search_alpha=pm_cfg.get("search_alpha", 1.0),
    )
    neural_points.eval()

    neural_points.refine_instance_ids_by_graph(distance=0.02, majority_ratio=0.8)

    points = neural_points.map_points.detach().cpu().numpy()
    features = neural_points.map_features.detach().cpu().numpy()
    instance_ids = neural_points.map_instance_ids.cpu().numpy()

    with h5py.File(output_path, "w") as hf:
        write_neural_points_to_hdf5(
            hf, points, features, instance_ids,
            instance_id_map, exclude_ids=exclude_ids,
        )

    print(f"  -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export neural point maps to HDF5.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory (contains config.yaml, neural_points/, instance_id_to_name.json).")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for HDF5 files (default: <run_root>/exported_neural_points/task-XXXX).")
    parser.add_argument("--split", type=str, default="eval", help="Subdirectory under neural_points/ (default: train). Use 'all' to export both train and eval.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default: cpu).")
    parser.add_argument("--exclude_categories", type=str, nargs="*", default=["background", "wall", "floors", "ceiling", "roof", "door", "window", "fence", "lawn", "electric"], help="Categories to exclude (default: background wall ceiling roof fence lawn).")
    args = parser.parse_args()

    # --- Load model config --- #
    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # --- Load instance ID map --- #
    id_map_path = os.path.join(args.run_dir, "instance_id_to_name.json")
    instance_id_map = {}
    if os.path.exists(id_map_path):
        instance_id_map = load_instance_id_map_from_json(id_map_path)
        print(f"[INFO] Loaded instance ID map ({len(instance_id_map)} entries)")
    else:
        print(f"[WARN] instance_id_to_name.json not found in {args.run_dir}")

    # --- Compute excluded instance IDs --- #
    exclude_ids: set[int] = set()
    if instance_id_map and args.exclude_categories:
        exclude_ids = filter_ids_by_keywords(instance_id_map, args.exclude_categories)
        print(f"[INFO] Excluding {len(exclude_ids)} instances matching categories: {args.exclude_categories}")

    # --- Output directory --- #
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_exported_neural_points_task_dir(args.run_dir)

    # --- Resolve splits --- #
    splits = ["train", "eval"] if args.split == "all" else [args.split]

    for split in splits:
        # --- Find .pt files --- #
        neural_points_dir = os.path.join(args.run_dir, "neural_points", split)
        if not os.path.isdir(neural_points_dir):
            # Fallback: try neural_points/ directly (no split subdirectory)
            neural_points_dir = os.path.join(args.run_dir, "neural_points")

        pt_files = sorted(
            f for f in os.listdir(neural_points_dir) if f.endswith(".pt")
        )
        if not pt_files:
            print(f"[WARN] No .pt files found in {neural_points_dir}, skipping split '{split}'")
            continue

        print(f"[INFO] Found {len(pt_files)} .pt files in {neural_points_dir}")

        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        # --- Export each file --- #
        for pt_file in pt_files:
            episode_name = os.path.splitext(pt_file)[0]
            pt_path = os.path.join(neural_points_dir, pt_file)
            output_path = os.path.join(split_output_dir, f"{episode_name}.hdf5")
            print(f"[EXPORT] {pt_file}")
            export_single(pt_path, output_path, model_config, instance_id_map, exclude_ids=exclude_ids, device=args.device)

        print(f"[DONE] Exported {len(pt_files)} files to {split_output_dir}")


if __name__ == "__main__":
    main()
