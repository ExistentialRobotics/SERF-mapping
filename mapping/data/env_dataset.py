import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from pathlib import Path
import h5py

from mapping.utils.category_utils import build_instance_to_category

__all__ = ["EnvDataset", "env_collate_fn", "EnvBatchSampler", "env_worker_init_fn"]



class EnvDataset(Dataset):
    """
    HDF5-based dataset for loading RGB, depth, poses, and optional DINO features.

    Expected HDF5 structure:
    - rgb: (N, H, W, 3) uint8
    - depth: (N, H, W) uint16 (mm)
    - poses: (N, 4, 4) float64 - world-to-camera (OpenCV convention)
    - intrinsics: (3, 3) float64 - camera intrinsic matrix
    - dino: (N, C, Hf, Wf) float32 (optional) - pre-computed DINO features
    - seg_instance_id: (N, H, W) int16 (optional) - instance segmentation

    Note: File handles are cached for performance. Call close() when done.
    """

    def __init__(self, dataset_dir, target_envs, num_images, image_size, patch_size, load_dino_features=False):
        # File handle cache for faster access
        self._file_cache = {}
        self.image_size = image_size
        self.patch_size = patch_size
        self.feat_h = self.image_size // patch_size
        self.load_dino_features = load_dino_features

        self.samples = []
        self.cam_to_world_poses = {}
        self.env_names = []
        self.instance_id_to_name = {}  # shared id->name mapping (validated across all episodes)
        self.excluded_instance_ids = set()  # instance IDs to exclude (glass, mirror, etc.)
        self.instance_to_category = {}  # shared: {inst_id: category}

        # Pre-loaded data: depth/seg/sam resized to feat_h resolution
        self._preloaded = {}  # {hdf5_path_str: {"depth": np.array, "seg": np.array, "sam": np.array}}

        dataset_dir = Path(dataset_dir)

        if not target_envs:
            # Look for HDF5 files
            hdf5_files = sorted(list(dataset_dir.rglob("*.hdf5")))
        else:
            # Check if target_envs are paths to HDF5 files or task directories
            hdf5_files = []
            for env in target_envs:
                path = dataset_dir / env
                if path.is_file() and path.suffix == ".hdf5":
                    hdf5_files.append(path)
                elif path.is_dir():
                    hdf5_files.extend(sorted(list(path.glob("*.hdf5"))))

        print(f"\nFound {len(hdf5_files)} HDF5 files.")
        self.hdf5_files = hdf5_files

        # --- Validate instance_id_to_name consistency across all episodes --- #
        reference_id_to_name = None
        reference_file = None
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                if 'instance_id_to_name' not in f.attrs:
                    continue
                id_to_name = json.loads(f.attrs['instance_id_to_name'])
                if reference_id_to_name is None:
                    reference_id_to_name = id_to_name
                    reference_file = hdf5_path
                elif id_to_name != reference_id_to_name:
                    # Find differences for error message
                    ref_keys = set(reference_id_to_name.keys())
                    cur_keys = set(id_to_name.keys())
                    only_in_ref = ref_keys - cur_keys
                    only_in_cur = cur_keys - ref_keys
                    diff_values = {k for k in ref_keys & cur_keys if reference_id_to_name[k] != id_to_name[k]}
                    raise ValueError(
                        f"instance_id_to_name mismatch between episodes.\n"
                        f"  Reference: {reference_file.name}\n"
                        f"  Mismatch:  {hdf5_path.name}\n"
                        f"  IDs only in reference: {only_in_ref}\n"
                        f"  IDs only in current:   {only_in_cur}\n"
                        f"  IDs with different names: {diff_values}"
                    )

        # Build shared mapping from the validated instance_id_to_name
        if reference_id_to_name:
            self.instance_id_to_name = reference_id_to_name
            self.instance_to_category = build_instance_to_category(reference_id_to_name)
            for inst_id, name in reference_id_to_name.items():
                name_lower = name.lower()
                if '/glass/' in name_lower or 'mirror' in name_lower:
                    self.excluded_instance_ids.add(int(inst_id))
            if self.excluded_instance_ids:
                print(f"[INFO] Found {len(self.excluded_instance_ids)} transparent/reflective instances to exclude: {sorted(self.excluded_instance_ids)}")
            unique_cats = set(self.instance_to_category.values())
            print(f"[INFO] Found {len(unique_cats)} unique object categories")

        for hdf5_path in self.hdf5_files:
            env_name = hdf5_path.stem  # episode_00090010
            # Include task name to be unique: task-0009_episode_00090010
            unique_env_name = f"{hdf5_path.parent.name}_{env_name}"

            print(f"--- Processing environment: {unique_env_name} ({hdf5_path.name}) ---")

            with h5py.File(hdf5_path, 'r') as f:
                if "depth" not in f or "poses" not in f:
                    print(f"  [WARN] Missing depth or poses in {hdf5_path}. Skipping.")
                    continue

                num_frames = f["depth"].shape[0]

                # Load poses (N, 4, 4) world-to-camera (OpenCV convention)
                # Convert to cam_to_world_poses
                poses_w2c = f["poses"][:]  # (N, 4, 4)
                poses_c2w = np.linalg.inv(poses_w2c)
                self.cam_to_world_poses[unique_env_name] = poses_c2w

                # Check for DINO
                has_dino = "dino" in f
                if self.load_dino_features and not has_dino:
                    print(f"  [WARN] load_dino_features=True but 'dino' dataset missing in {hdf5_path}. Will compute on fly (if model loaded) or fail.")

                # Check for SAM masks
                has_sam = "sam_masks" in f
                if has_sam:
                    print(f"  [INFO] SAM masks available ({f['sam_masks'].shape[0]} frames)")

                # Indices to use
                indices = range(num_frames)
                if num_images > 0:
                    indices = indices[:num_images]
                n_load = len(indices)

                # --- Pre-load depth/seg/sam at feat_h resolution --- #
                hdf5_key = str(hdf5_path)
                preloaded = {}
                batch_load = 1000

                # Depth → float32 meters at (feat_h, feat_h)
                chunks = []
                for s in range(0, n_load, batch_load):
                    e = min(s + batch_load, n_load)
                    raw = f['depth'][s:e].astype(np.float32) / 1000.0
                    t = torch.from_numpy(raw).unsqueeze(1)
                    t = F.interpolate(t, (self.feat_h, self.feat_h), mode="nearest-exact").squeeze(1)
                    chunks.append(t.numpy())
                preloaded["depth"] = np.concatenate(chunks, axis=0)

                # Segmentation → int32 at (feat_h, feat_h)
                if 'seg_instance_id' in f:
                    chunks = []
                    for s in range(0, n_load, batch_load):
                        e = min(s + batch_load, n_load)
                        raw = f['seg_instance_id'][s:e].astype(np.float32)
                        t = torch.from_numpy(raw).unsqueeze(1)
                        t = F.interpolate(t, (self.feat_h, self.feat_h), mode="nearest-exact").squeeze(1)
                        chunks.append(t.to(torch.int32).numpy())
                    preloaded["seg"] = np.concatenate(chunks, axis=0)

                # SAM masks → int32 at (feat_h, feat_h)
                if has_sam:
                    chunks = []
                    for s in range(0, n_load, batch_load):
                        e = min(s + batch_load, n_load)
                        raw = f['sam_masks'][s:e].astype(np.float32)
                        t = torch.from_numpy(raw).unsqueeze(1)
                        t = F.interpolate(t, (self.feat_h, self.feat_h), mode="nearest-exact").squeeze(1)
                        chunks.append(t.to(torch.int32).numpy())
                    preloaded["sam"] = np.concatenate(chunks, axis=0)

                self._preloaded[hdf5_key] = preloaded
                mem_mb = sum(v.nbytes for v in preloaded.values()) / 1024 / 1024
                print(f"  [PRELOAD] Cached depth/seg/sam at {self.feat_h}x{self.feat_h}: {mem_mb:.1f} MB ({n_load} frames)")

                for idx in indices:
                    self.samples.append({
                        "hdf5_path": hdf5_key,
                        "idx": idx,
                        "env_name": unique_env_name,
                        "has_dino": has_dino,
                        "has_sam": has_sam,
                    })

            self.env_names.append(unique_env_name)

        print(f"\nDataset initialized with {len(self.samples)} total samples from {len(self.env_names)} environments.")

    def _get_file(self, hdf5_path: str):
        """Get cached file handle or open new one. Returns None if file cannot be opened."""
        if hdf5_path not in self._file_cache:
            if not Path(hdf5_path).exists():
                return None
            self._file_cache[hdf5_path] = h5py.File(hdf5_path, 'r')
        return self._file_cache[hdf5_path]

    def close(self):
        """Close all cached file handles."""
        for f in self._file_cache.values():
            f.close()
        self._file_cache.clear()

    def __del__(self):
        """Cleanup file handles on deletion."""
        self.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        hdf5_path = sample_info["hdf5_path"]
        frame_idx = sample_info["idx"]
        env_name = sample_info["env_name"]
        has_dino = sample_info["has_dino"]
        has_sam = sample_info.get("has_sam", False)

        preloaded = self._preloaded.get(hdf5_path)
        if preloaded is None:
            return None

        # Depth: already (feat_h, feat_h) float32 meters
        depth_t = torch.from_numpy(preloaded["depth"][frame_idx])

        # Pose
        E_cv = self.cam_to_world_poses[env_name][frame_idx][:3, :]
        extrinsic_t = torch.from_numpy(E_cv).float()

        # DINO: only remaining HDF5 read per frame (lzf, per-frame chunked → fast)
        dino_feat = None
        if self.load_dino_features and has_dino:
            f = self._get_file(hdf5_path)
            if f is not None and 'dino' in f:
                dino_feat = torch.from_numpy(f['dino'][frame_idx].astype(np.float32))

        # Segmentation: already (feat_h, feat_h) int32
        seg_inst = None
        if "seg" in preloaded:
            seg_inst = torch.from_numpy(preloaded["seg"][frame_idx]).long()

        # SAM mask: already (feat_h, feat_h) int32
        sam_mask = None
        if has_sam and "sam" in preloaded:
            sam_mask = torch.from_numpy(preloaded["sam"][frame_idx]).long()

        return {
            "depth_t": depth_t,
            "extrinsic_t": extrinsic_t,
            "env_name": env_name,
            "dino_feat": dino_feat,
            "seg_inst": seg_inst,
            "sam_mask": sam_mask,
            "excluded_instance_ids": self.excluded_instance_ids,
            "inst_to_category": self.instance_to_category,
        }


def env_collate_fn(batch):
    """
    Custom collate function to filter out None values from the batch.
    This is used to handle cases where `__getitem__` returns None for invalid samples.
    Also handles non-tensor fields like window_instance_ids (sets).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}

    # Separate tensor and non-tensor fields
    tensor_keys = []
    non_tensor_keys = []
    for key in batch[0].keys():
        if isinstance(batch[0][key], (torch.Tensor, np.ndarray)) or batch[0][key] is None:
            tensor_keys.append(key)
        else:
            non_tensor_keys.append(key)

    # Collate tensor fields using default_collate
    tensor_batch = [{k: sample[k] for k in tensor_keys} for sample in batch]
    result = torch.utils.data.dataloader.default_collate(tensor_batch)

    # Keep non-tensor fields as lists
    for key in non_tensor_keys:
        result[key] = [sample[key] for sample in batch]

    return result


def env_worker_init_fn(worker_id: int):
    """Clear stale h5py file handles after fork. Workers re-open on demand via _get_file."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset._file_cache.clear()


class EnvBatchSampler(Sampler):
    """
    BatchSampler that ensures each batch contains samples from only one environment.
    Samples within each env are shuffled, and env order is also shuffled each epoch.
    """

    def __init__(self, dataset: EnvDataset, batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group sample indices by env_name
        self.env_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.env_to_indices[sample["env_name"]].append(idx)

        self.env_names = list(self.env_to_indices.keys())

    def __iter__(self):
        # Shuffle env order
        env_order = np.random.permutation(self.env_names).tolist()

        for env_name in env_order:
            indices = self.env_to_indices[env_name].copy()
            np.random.shuffle(indices)

            # Yield batches for this env
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for indices in self.env_to_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total += n_batches
        return total
