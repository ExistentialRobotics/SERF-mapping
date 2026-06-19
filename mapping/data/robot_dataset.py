"""Dataset and sampler for robot hemisphere captures with DINO features.

RobotDataset pre-loads depth and robot mask at DINO feature resolution.
RobotBatchSampler groups frames by robot state for shared FK computation per batch.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class RobotDataset(Dataset):
    """Dataset for robot hemisphere captures with DINO features.

    Pre-loads depth and robot mask at DINO feature resolution.
    DINO features are read per-frame (lzf-compressed, fast random access).

    Args:
        hdf5_path: Path to the hemisphere HDF5 file (with 'dino' dataset).
        image_size: Original image size (for intrinsics scaling).
        patch_size: DINO patch size (feature resolution = image_size // patch_size).
    """

    def __init__(
        self,
        hdf5_path: str,
        image_size: int = 480,
        patch_size: int = 16,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.image_size = image_size
        self.patch_size = patch_size
        self.feat_h = image_size // patch_size
        self.feat_w = image_size // patch_size

        # File handle (re-opened per worker via worker_init_fn)
        self._file: Optional[h5py.File] = None

        with h5py.File(self.hdf5_path, "r") as f:
            self.n_frames = f["rgb"].shape[0]
            self.H_orig = f["rgb"].shape[1]
            self.W_orig = f["rgb"].shape[2]

            # Intrinsics (3, 3) -> fx, fy, cx, cy
            K = f["intrinsics"][:]
            self.fx = float(K[0, 0])
            self.fy = float(K[1, 1])
            self.cx = float(K[0, 2])
            self.cy = float(K[1, 2])

            # State metadata
            self.state_index = f["state_index"][:].astype(np.int32)  # (T,)
            self.n_states = int(f.attrs["n_states"])
            self.urdf_cfgs = f["robot_urdf_cfg"][:].astype(np.float32)  # (n_states, 28)
            self.base_transforms = f["robot_base_transform"][:].astype(np.float32)  # (n_states, 4, 4)

            # Poses (T, 4, 4) — extrinsic CV (world-to-camera)
            poses_w2c = f["poses"][:].astype(np.float32)
            self.poses_c2w = np.linalg.inv(poses_w2c).astype(np.float32)

            # DINO feature dim
            if "dino" not in f:
                raise ValueError(f"'dino' dataset not found in {hdf5_path}")
            self.dino_dim = f["dino"].shape[1]

            # Pre-load depth + seg at feature resolution
            self._preload_depth_and_mask(f)

    def _preload_depth_and_mask(self, f: h5py.File) -> None:
        """Pre-load depth, robot mask, and seg IDs at DINO feature resolution.

        Reads from HDF5 in chunks to avoid loading full-resolution data
        (18000x480x480) entirely into RAM (~60 GB peak).
        """
        T = f["depth"].shape[0]
        chunk_size = 1000

        # Pre-allocate output tensors at feature resolution (~100 MB total)
        self.depth_feat = torch.zeros(T, self.feat_h, self.feat_w, dtype=torch.float32)
        self.robot_mask_feat = torch.zeros(T, self.feat_h, self.feat_w, dtype=torch.bool)
        self.seg_feat = torch.zeros(T, self.feat_h, self.feat_w, dtype=torch.int32)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)

            # Read chunk from HDF5 (only this chunk in RAM at full resolution)
            depth_chunk = f["depth"][start:end]  # (C, H, W) uint16
            seg_chunk = f["seg_instance_id"][start:end]  # (C, H, W) int16

            # Convert and downsample depth
            depth_m = torch.from_numpy(depth_chunk.astype(np.float32)).unsqueeze(1) / 1000.0
            d = F.interpolate(depth_m, size=(self.feat_h, self.feat_w), mode="nearest-exact")
            self.depth_feat[start:end] = d.squeeze(1)

            # Convert and downsample robot mask
            mask = torch.from_numpy((seg_chunk > 0).astype(np.float32)).unsqueeze(1)
            m = F.interpolate(mask, size=(self.feat_h, self.feat_w), mode="nearest-exact")
            self.robot_mask_feat[start:end] = m.squeeze(1) > 0.5

            # Convert and downsample seg IDs
            seg_f = torch.from_numpy(seg_chunk.astype(np.float32)).unsqueeze(1)
            s = F.interpolate(seg_f, size=(self.feat_h, self.feat_w), mode="nearest-exact")
            self.seg_feat[start:end] = s.squeeze(1).to(torch.int32)

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._get_file()
        dino_feat = torch.from_numpy(f["dino"][idx].astype(np.float32))  # (C, Hf, Wf)

        return {
            "depth": self.depth_feat[idx],                                   # (Hf, Wf)
            "dino_feat": dino_feat,                                          # (C, Hf, Wf)
            "extrinsic": torch.from_numpy(self.poses_c2w[idx][:3, :]),       # (3, 4)
            "robot_mask": self.robot_mask_feat[idx],                         # (Hf, Wf) bool
            "seg_inst": self.seg_feat[idx],                                  # (Hf, Wf) int32
            "state_index": self.state_index[idx],                            # int32
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class RobotBatchSampler(Sampler):
    """Batch sampler that groups frames by robot state index.

    All frames in a batch share the same robot state, enabling
    shared FK computation per batch.
    """

    def __init__(
        self,
        state_index: np.ndarray,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        # Group frame indices by state
        self.state_groups: Dict[int, List[int]] = {}
        for idx, si in enumerate(state_index):
            si = int(si)
            if si not in self.state_groups:
                self.state_groups[si] = []
            self.state_groups[si].append(idx)

    def __iter__(self):
        state_ids = list(self.state_groups.keys())
        if self.shuffle:
            self.rng.shuffle(state_ids)

        for si in state_ids:
            indices = list(self.state_groups[si])
            if self.shuffle:
                self.rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.state_groups.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total += n_batches
        return total


def robot_collate_fn(batch):
    """Custom collate function that filters None and handles non-tensor fields."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}

    tensor_keys = []
    non_tensor_keys = []
    for key in batch[0].keys():
        if isinstance(batch[0][key], (torch.Tensor, np.ndarray)) or batch[0][key] is None:
            tensor_keys.append(key)
        else:
            non_tensor_keys.append(key)

    tensor_batch = [{k: sample[k] for k in tensor_keys} for sample in batch]
    result = torch.utils.data.dataloader.default_collate(tensor_batch)

    for key in non_tensor_keys:
        result[key] = [sample[key] for sample in batch]

    return result


def robot_worker_init_fn(worker_id: int) -> None:
    """Re-open HDF5 file handle per DataLoader worker."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._file = None  # Force re-open in worker process
