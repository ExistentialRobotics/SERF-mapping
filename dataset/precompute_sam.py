import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

warnings.filterwarnings("ignore", message=".*undefined symbol.*")

# Add sam2 submodule to path
SAM2_PATH = Path(__file__).resolve().parent.parent / "external" / "sam2"
if SAM2_PATH.exists() and str(SAM2_PATH) not in sys.path:
    sys.path.insert(0, str(SAM2_PATH))

# Model configurations
SAM2_MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


def load_sam2_mask_generator(
    model_size: str = "large",
    device: str = "cuda",
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.92,
    crop_n_layers: int = 1,
    min_mask_region_area: int = 100,
):

    if model_size not in SAM2_MODELS:
        raise ValueError(f"Unknown model: {model_size}. Choose from {list(SAM2_MODELS.keys())}")

    config_file, checkpoint_name = SAM2_MODELS[model_size]

    # Download checkpoint if needed
    checkpoint_dir = Path.home() / ".cache" / "sam2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name

    if not checkpoint_path.exists():
        print(f"Downloading SAM2 {model_size} checkpoint...")
        import urllib.request
        urllib.request.urlretrieve(f"{SAM2_BASE_URL}/{checkpoint_name}", checkpoint_path)
        print(f"Downloaded to {checkpoint_path}")

    sam2 = build_sam2(config_file, str(checkpoint_path), device=device)

    return SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=0.95,
        crop_n_layers=crop_n_layers,
        box_nms_thresh=0.7,
        min_mask_region_area=min_mask_region_area,
    )


def masks_to_ids(masks: list, shape: tuple) -> np.ndarray:
    """Convert SAM2 masks to ID map. Larger masks are painted first, smaller override."""
    if not masks:
        return np.full(shape, -1, dtype=np.int16)

    mask_map = np.full(shape, -1, dtype=np.int16)
    for idx, m in enumerate(sorted(masks, key=lambda x: x["area"], reverse=True)):
        mask_map[m["segmentation"]] = idx

    return mask_map


def generate_masks(mask_generator, image: np.ndarray) -> np.ndarray:
    """Generate mask IDs for a single image."""
    masks = mask_generator.generate(image)
    return masks_to_ids(masks, image.shape[:2])


def process_hdf5(hdf5_path: Path, mask_generator, overwrite: bool) -> int:
    """Process single HDF5 file and add SAM masks."""
    with h5py.File(hdf5_path, "r+") as f:
        if "rgb" not in f:
            print(f"  [SKIP] No RGB in {hdf5_path.name}")
            return 0

        if "sam_masks" in f and not overwrite:
            print(f"  [SKIP] Already processed: {hdf5_path.name}")
            return 0

        num_frames, H, W = f["rgb"].shape[:3]

        # Clear existing
        if "sam_masks" in f:
            del f["sam_masks"]

        # Create datasets
        sam_masks = f.create_dataset(
            "sam_masks", (num_frames, H, W), dtype=np.int16, compression="gzip", compression_opts=4
        )

        print(f"\n  Processing {hdf5_path.name} ({num_frames} frames, {H}x{W})")

        for idx in tqdm(range(num_frames), desc="  Frames", leave=True, dynamic_ncols=True):
            image = f["rgb"][idx]
            mask_ids = generate_masks(mask_generator, image)

            sam_masks[idx] = mask_ids

    return num_frames


def find_hdf5_files(dataset_path: Path, target_envs: list) -> list:
    """Find HDF5 files to process."""
    if not target_envs:
        return sorted(dataset_path.rglob("episode_*.hdf5"))

    files = []
    for env in target_envs:
        path = dataset_path / env
        if path.is_file() and path.suffix == ".hdf5":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("episode_*.hdf5")))
    return files


def main():
    parser = argparse.ArgumentParser(description="Precompute SAM2 masks for HDF5 dataset")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to a single HDF5 file (alternative to --dataset_dir)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Directory containing HDF5 files")
    parser.add_argument("--target_envs", type=str, nargs="*", default=[])
    parser.add_argument("--sam_model", type=str, default="base_plus", choices=SAM2_MODELS.keys())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=512)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.85)
    parser.add_argument("--stability_score_thresh", type=float, default=0.92)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--min_mask_region_area", type=int, default=120)
    args = parser.parse_args()

    # Determine HDF5 files to process
    if args.input_path:
        input_file = Path(args.input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        hdf5_files = [input_file]
    elif args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        hdf5_files = find_hdf5_files(dataset_path, args.target_envs)
    else:
        parser.error("Either --input_path or --dataset_dir is required")

    if not hdf5_files:
        print("No HDF5 files found!")
        return

    print(f"Found {len(hdf5_files)} HDF5 files")

    print(f"Loading SAM2 ({args.sam_model})...")
    mask_generator = load_sam2_mask_generator(
        model_size=args.sam_model,
        device=args.device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
    )
    print("SAM2 loaded!")

    total_frames = 0
    for hdf5_path in tqdm(hdf5_files, desc="Processing files"):
        total_frames += process_hdf5(hdf5_path, mask_generator, args.overwrite)

    print(f"\nDone! Processed {total_frames} frames across {len(hdf5_files)} files.")


if __name__ == "__main__":
    main()
