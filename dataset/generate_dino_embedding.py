import argparse
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import h5py
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapping.models.vision_backbone import DINOv3Wrapper


def get_robot_ids_from_attrs(attrs):
    if "instance_id_to_name" not in attrs:
        return []
    mapping = json.loads(attrs["instance_id_to_name"])
    robot_ids = []
    for str_id, name in mapping.items():
        name_lower = name.lower()
        if "robot" in name_lower:
            robot_ids.append(int(str_id))
    return robot_ids


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, transform=None, mask_robot=False):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.mask_robot = mask_robot
        self.robot_ids = []

        # Read metadata
        with h5py.File(hdf5_path, 'r') as f:
            if "rgb" not in f:
                raise ValueError(f"No 'rgb' dataset found in {hdf5_path}")
            self.length = f['rgb'].shape[0]

            if self.mask_robot:
                self.robot_ids = get_robot_ids_from_attrs(f.attrs)
                if self.robot_ids:
                    print(f"[INFO] Found robot IDs for masking: {self.robot_ids}")
                else:
                    print("[INFO] No robot IDs found for masking.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file in each worker/iteration to be safe with HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            rgb = f['rgb'][idx] # (H, W, 3) uint8

            if self.mask_robot and self.robot_ids:
                if 'seg_instance_id' in f:
                    seg = f['seg_instance_id'][idx] # (H, W)
                    mask = np.isin(seg, self.robot_ids)
                    rgb[mask] = 0
                else:
                    # Warn only once? Difficult in worker.
                    pass

        # RGB is uint8 (H, W, 3)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor


def load_model(args, device):
    repo_dir = Path("external/dinov3")
    if not repo_dir.exists():
        repo_dir = Path(__file__).resolve().parent.parent / "external" / "dinov3"

    if not repo_dir.exists():
        print(f"Error: Could not find external/dinov3 at ./external/dinov3 or {repo_dir}")
        return None

    print(f"Loading DINOv3 model from {repo_dir}...")
    print(f"Loading backbone weights from: {args.backbone_weights}")
    print(f"Loading DINOv3 feature weights from: {args.dinotxt_weights}")
    dinotxt_model, tokenizer = torch.hub.load(
        str(repo_dir),
        'dinov3_vitl16_dinotxt_tet1280d20h24l',
        source='local',
        backbone_weights=args.backbone_weights,
        weights=args.dinotxt_weights,
    )
    model = DINOv3Wrapper(dinotxt_model, tokenizer).to(device).eval()
    print(f"DINOv3 model loaded. Feature dim: {model.feature_dim}")
    return model


def write_dino_features(hdf5_file: Path, data: np.ndarray) -> None:
    with h5py.File(hdf5_file, 'r+') as f:
        if 'dino' in f:
            dset = f['dino']
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            dset[-data.shape[0]:] = data
        else:
            maxshape = (None,) + data.shape[1:]
            f.create_dataset(
                "dino",
                data=data,
                maxshape=maxshape,
                chunks=(1,) + data.shape[1:],
                compression="lzf"
            )


def process_hdf5_file(hdf5_file: Path, args, model, transform, device) -> bool:
    with h5py.File(hdf5_file, 'r') as f:
        if 'dino' in f:
            if not args.overwrite:
                print(f"Skipping {hdf5_file.name}: 'dino' dataset already exists (use --overwrite to replace).")
                return False

    if args.overwrite:
        with h5py.File(hdf5_file, 'r+') as f:
            if 'dino' in f:
                del f['dino']

    dataset = HDF5Dataset(hdf5_file, transform=transform, mask_robot=args.mask_robot)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    buffer = []

    for images in tqdm(dataloader, desc=f"Inference {hdf5_file.name}"):
        images = images.to(device)
        with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            # model returns (B, C, Hf, Wf)
            features = model(images).cpu().numpy()
            buffer.append(features)

        if len(buffer) >= args.flush_threshold:
            data_to_write = np.concatenate(buffer, axis=0)
            buffer = []
            write_dino_features(hdf5_file, data_to_write)

    if buffer:
        data_to_write = np.concatenate(buffer, axis=0)
        write_dino_features(hdf5_file, data_to_write)

    print(f"Done: {hdf5_file}")
    return True


def resolve_hdf5_inputs(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    if input_path.is_file():
        if input_path.suffix.lower() != ".hdf5":
            raise ValueError(f"Input file {input_path} is not an HDF5 file.")
        return [input_path]

    if input_path.is_dir():
        hdf5_files = sorted(path for path in input_path.rglob("*.hdf5") if path.is_file())
        if not hdf5_files:
            raise ValueError(f"No .hdf5 files found under {input_path}.")
        return hdf5_files

    raise ValueError(f"Input path {input_path} is neither a file nor a directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DINO embeddings for a single HDF5 file or a directory of HDF5 files."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a single .hdf5 file or a directory to scan recursively, e.g. data/mapping_dataset.",
    )
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="/mnt/dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="Path or URL to DINOv3 backbone weights.",
    )
    parser.add_argument(
        "--dinotxt_weights",
        type=str,
        default="/mnt/dino/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        help="Path or URL to DINOv3 feature weights.",
    )
    parser.add_argument("--image_size", type=int, default=480, help="Image size for DINO.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading. Default 0 for HDF5 safety.",
    )
    parser.add_argument("--mask_robot", action="store_true", help="Whether to mask out the robot in the input images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing 'dino' dataset in HDF5.")
    parser.add_argument(
        "--flush_threshold",
        type=int,
        default=50,
        help="Number of batches to buffer before flushing to disk.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    try:
        hdf5_files = resolve_hdf5_inputs(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    if len(hdf5_files) == 1:
        print(f"Found 1 HDF5 file: {hdf5_files[0]}")
    else:
        print(f"Found {len(hdf5_files)} HDF5 files under {input_path}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args, device)
    if model is None:
        return

    transform = transforms.Compose([
        transforms.Resize(
            (args.image_size, args.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ])

    processed_count = 0
    skipped_count = 0
    failures = []

    for index, hdf5_file in enumerate(hdf5_files, start=1):
        print(f"\n[{index}/{len(hdf5_files)}] Processing {hdf5_file}")
        try:
            processed = process_hdf5_file(hdf5_file, args, model, transform, device)
        except Exception as exc:
            failures.append((hdf5_file, exc))
            print(f"Error processing {hdf5_file}: {exc}")
            continue

        if processed:
            processed_count += 1
        else:
            skipped_count += 1

    print(
        f"\nDINO embedding generation complete: "
        f"{processed_count} processed, {skipped_count} skipped, {len(failures)} failed."
    )
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
