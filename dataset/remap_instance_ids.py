import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from tqdm import tqdm

# Try to import torch for GPU acceleration
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, using CPU")


def find_hdf5_files(input_dir: str) -> list[str]:
    """Find HDF5 files under ``input_dir`` recursively."""
    return sorted(glob.glob(os.path.join(input_dir, "**", "*.hdf5"), recursive=True))


def load_instance_map_for_generation(hdf5_path: str) -> dict[str, str]:
    """
    Load the best available instance map for fixed-map generation.

    Prefer ``original_instance_id_to_name`` so reruns remain stable even after
    some files have already been remapped in-place.
    """
    with h5py.File(hdf5_path, "r") as f:
        if "original_instance_id_to_name" in f.attrs:
            return json.loads(f.attrs["original_instance_id_to_name"])
        if "instance_id_to_name" in f.attrs:
            return json.loads(f.attrs["instance_id_to_name"])
    return {}


def build_fixed_map_from_names(instance_names: Iterable[str]) -> dict[str, int]:
    """Build a deterministic name->id mapping by sorting names alphabetically."""
    return {name: idx for idx, name in enumerate(sorted(set(instance_names)))}


def build_fixed_map_from_hdf5_files(hdf5_files: list[str]) -> dict[str, int]:
    """Aggregate instance names from HDF5 attrs and build a sorted fixed map."""
    all_names: list[str] = []
    for hdf5_path in hdf5_files:
        instance_map = load_instance_map_for_generation(hdf5_path)
        all_names.extend(instance_map.values())

    fixed_map = build_fixed_map_from_names(all_names)
    if not fixed_map:
        raise ValueError("Could not build instance ID map: no instance_id_to_name attrs were found.")
    return fixed_map


def generate_and_save_fixed_map(hdf5_files: list[str], output_path: str) -> dict[str, int]:
    """Build a folder-wide fixed map and save it as JSON."""
    fixed_map = build_fixed_map_from_hdf5_files(hdf5_files)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fixed_map, f, indent=2, sort_keys=True)
    return fixed_map


def load_or_generate_fixed_map(hdf5_files: list[str], instance_id_map_path: str) -> dict[str, int]:
    """Load an existing fixed map when present, otherwise generate and save one."""
    if os.path.exists(instance_id_map_path):
        with open(instance_id_map_path, "r") as f:
            return json.load(f)
    return generate_and_save_fixed_map(hdf5_files, instance_id_map_path)


def get_default_instance_id_map_path(input_dir: str) -> str:
    """
    Resolve the default output path for the generated fixed instance ID map.

    For shared PVC layouts like:
      /mnt/4d_latent_mapping/demonstration_replay/task-0021

    the compatible output path is:
      /mnt/4d_latent_mapping/instance_id_map/task-0021/instance_id_map.json

    Otherwise fall back to writing inside the input directory.
    """
    input_path = Path(input_dir)
    if input_path.name.startswith("task-"):
        root_dir = input_path.parent.parent
        return str(root_dir / "instance_id_map" / input_path.name / "instance_id_map.json")
    return str(input_path / "instance_id_map.json")


def remap_segmentation_gpu(seg_data: np.ndarray, original_to_fixed: dict) -> np.ndarray:
    """
    Remap segmentation IDs using GPU (PyTorch).
    """
    seg_tensor = torch.from_numpy(seg_data.astype(np.int64)).cuda()
    result = torch.zeros_like(seg_tensor, dtype=torch.int64)

    for orig_id, fixed_id in original_to_fixed.items():
        mask = seg_tensor == orig_id
        result[mask] = fixed_id

    return result.cpu().numpy().astype(np.uint16)


def remap_segmentation(seg_data: np.ndarray, original_map: dict, fixed_map: dict) -> np.ndarray:
    """
    Remap segmentation IDs from original to fixed mapping.
    """
    original_to_fixed = {}
    for orig_id_str, obj_name in original_map.items():
        orig_id = int(orig_id_str)
        if obj_name in fixed_map:
            original_to_fixed[orig_id] = fixed_map[obj_name]
        else:
            original_to_fixed[orig_id] = 65535

    if HAS_TORCH:
        return remap_segmentation_gpu(seg_data, original_to_fixed)

    remapped_seg = np.zeros_like(seg_data, dtype=np.uint16)
    for orig_id, fixed_id in original_to_fixed.items():
        remapped_seg[seg_data == orig_id] = fixed_id

    return remapped_seg


def read_with_retry(dset, start_idx, end_idx, max_retries=3, delay=1.0):
    """
    Read HDF5 dataset with retry logic for intermittent NFS errors.
    """
    for attempt in range(max_retries):
        try:
            return dset[start_idx:end_idx]
        except OSError as e:
            if attempt < max_retries - 1:
                print(f"    Read error (attempt {attempt+1}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise e


def iter_segmentation_containers(hdf5_file: h5py.File, dataset_keys: tuple[str, ...]) -> list:
    """Return root/group containers that directly own a segmentation dataset."""
    containers = []
    seen_paths = set()

    def maybe_add(container) -> None:
        for dataset_key in dataset_keys:
            if dataset_key in container and isinstance(container[dataset_key], h5py.Dataset):
                if container.name not in seen_paths:
                    seen_paths.add(container.name)
                    containers.append(container)
                return

    maybe_add(hdf5_file)

    def visitor(_name: str, obj) -> None:
        if isinstance(obj, h5py.Group):
            maybe_add(obj)

    hdf5_file.visititems(visitor)
    return containers


def process_hdf5_file(hdf5_path: str, fixed_map: dict) -> bool:
    """
    Process a single HDF5 file to remap instance IDs in-place.

    Strategy:
    1. Create temp dataset 'seg_instance_id_tmp'
    2. Read from original, remap, write to tmp
    3. If successful, delete original and rename tmp

    Returns:
        True on success, False on error
    """
    tmp_key = "seg_instance_id_tmp"

    try:
        # Open file in read/write mode
        with h5py.File(hdf5_path, "r+") as f:
            if "instance_id_to_name" not in f.attrs:
                print(f"  [SKIP] No instance_id_to_name: {os.path.basename(hdf5_path)}")
                return False

            current_map = json.loads(f.attrs["instance_id_to_name"])
            fixed_id_to_name = {str(v): k for k, v in fixed_map.items()}

            if current_map == fixed_id_to_name:
                if "original_instance_id_to_name" not in f.attrs:
                    f.attrs["original_instance_id_to_name"] = json.dumps(current_map)
                return True

            containers = iter_segmentation_containers(f, ("seg_instance_id",))
            if not containers:
                print(f"  [SKIP] No seg_instance_id dataset: {os.path.basename(hdf5_path)}")
                return False

            # Process each container that owns a seg_instance_id dataset.
            for container in containers:
                orig_dset = container["seg_instance_id"]
                n_frames, h, w = orig_dset.shape

                # Clean up any leftover tmp dataset
                if tmp_key in container:
                    del container[tmp_key]

                # Create tmp dataset with same properties
                tmp_dset = container.create_dataset(
                    tmp_key,
                    shape=(n_frames, h, w),
                    dtype=np.uint16,
                    chunks=orig_dset.chunks,
                    compression=orig_dset.compression,
                    compression_opts=orig_dset.compression_opts,
                )

                # Process in chunks
                chunk_size = 100
                for start_idx in range(0, n_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_frames)

                    # Read with retry for NFS issues
                    seg_chunk = read_with_retry(orig_dset, start_idx, end_idx)

                    # Remap
                    remapped_chunk = remap_segmentation(seg_chunk, current_map, fixed_map)

                    # Write to tmp
                    tmp_dset[start_idx:end_idx] = remapped_chunk

                # Success - swap datasets
                del container["seg_instance_id"]
                container.move(tmp_key, "seg_instance_id")

            # Update attrs
            if "original_instance_id_to_name" not in f.attrs:
                f.attrs["original_instance_id_to_name"] = json.dumps(current_map)
            f.attrs["instance_id_to_name"] = json.dumps(fixed_id_to_name)

        return True

    except Exception as e:
        print(f"  [ERROR] {os.path.basename(hdf5_path)}: {e}")
        # Try to clean up tmp dataset
        try:
            with h5py.File(hdf5_path, "r+") as f:
                for container in iter_segmentation_containers(f, ("seg_instance_id", tmp_key)):
                    if tmp_key in container:
                        del container[tmp_key]
        except Exception:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(description="Remap instance IDs in HDF5 files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing HDF5 files to process")
    parser.add_argument(
        "--instance_id_map",
        type=str,
        default=None,
        help="Path to the fixed instance ID mapping JSON (name -> id). If the file exists it is loaded and reused; otherwise a new fixed map is generated there. Defaults to <root>/instance_id_map/task-XXXX/instance_id_map.json when input_dir is a task folder, otherwise <input_dir>/instance_id_map.json",
    )
    args = parser.parse_args()

    # Find all HDF5 files
    hdf5_files = find_hdf5_files(args.input_dir)
    if not hdf5_files:
        print(f"No HDF5 files found in {args.input_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files to process")

    instance_id_map_path = args.instance_id_map or get_default_instance_id_map_path(args.input_dir)
    fixed_map_exists = os.path.exists(instance_id_map_path)
    fixed_map = load_or_generate_fixed_map(hdf5_files, instance_id_map_path)
    if fixed_map_exists:
        print(f"Loaded fixed instance ID mapping with {len(fixed_map)} entries from {instance_id_map_path}")
    else:
        print(f"Generated fixed instance ID mapping with {len(fixed_map)} entries")
        print(f"Saved generated instance ID map to {instance_id_map_path}")

    # Process each file
    success_count = 0
    error_count = 0

    for hdf5_path in tqdm(hdf5_files, desc="Processing files"):
        result = process_hdf5_file(hdf5_path, fixed_map)
        if result:
            success_count += 1
        else:
            error_count += 1

    print(f"\nDone! Processed: {success_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
