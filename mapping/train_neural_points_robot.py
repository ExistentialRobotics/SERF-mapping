"""Train a single neural point map for the robot across multiple joint configurations.

Uses RobotSurfaceSampler for FK-based point correspondence across robot states,
and learns per-surface-point latent features decoded to DINO embeddings.

Usage:
    python mapping/train_neural_points_robot.py \
        --config mapping/config/config_robot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from mapping.models.mlp import MLP
from mapping.losses.cosine_similarity import cosine_similarity_loss
from mapping.losses.inter_category_loss import compute_infonce_loss
from mapping.representations.robot_neural_points import RobotNeuralPointMap
from mapping.data.robot_dataset import (
    RobotDataset,
    RobotBatchSampler,
    robot_worker_init_fn,
)
from utils.geometry import unproject_depth_to_world
from utils.robot import URDF_PATH, RobotSurfaceSampler
from mapping.config.robot_train_config import RobotTrainConfig
from mapping.utils.logger import TrainLogger


# --------------------------------------------------------------------------- #
#  FK Pre-computation
# --------------------------------------------------------------------------- #

def precompute_surface_cache(
    sampler: RobotSurfaceSampler,
    urdf_cfgs: np.ndarray,
    base_transforms: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Pre-compute FK surface points for all robot states.

    Args:
        sampler: Robot surface point sampler.
        urdf_cfgs: (n_states, N_URDF_JOINTS) joint configs.
        base_transforms: (n_states, 4, 4) base transforms.
        device: Target device.

    Returns:
        (n_states, N_surface, 3) float32 tensor on device.
    """
    n_states = urdf_cfgs.shape[0]
    n_pts = sampler.n_total_points
    cache = torch.zeros(n_states, n_pts, 3, dtype=torch.float32)

    for si in range(n_states):
        pts, _ = sampler.get_points(urdf_cfgs[si], base_transforms[si])
        cache[si] = torch.from_numpy(pts)

    print(f"[FK] Pre-computed surface points for {n_states} states "
          f"({n_pts} points each, {cache.numel() * 4 / 1024:.0f} KB)")
    return cache.to(device)


# --------------------------------------------------------------------------- #
#  Training Step
# --------------------------------------------------------------------------- #

def train_step(
    data: Dict[str, torch.Tensor],
    robot_points: RobotNeuralPointMap,
    decoder: MLP,
    optimizer: torch.optim.Optimizer,
    surface_cache: torch.Tensor,
    intrinsics: Tuple[float, float, float, float, int, int],
    robot_losses: Dict[str, float],
    device: str,
    no_backward: bool = False,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Execute a single robot training step.

    Supports reconstruction loss and part-level contrastive loss.

    Args:
        data: Batch dict from RobotDataset.
        robot_points: Learnable surface features.
        decoder: Latent -> DINO decoder.
        optimizer: Optimizer for robot_points.features + decoder.parameters().
        surface_cache: (n_states, N, 3) pre-computed FK surface points.
        intrinsics: (fx, fy, cx, cy, H_orig, W_orig).
        robot_losses: Dict with keys lambda_contrastive, contrastive_temperature,
            contrastive_max_samples.
        device: Torch device string.
        no_backward: If True, skip backward/optimizer step and return the raw
            loss tensor as the 4th element for a combined backward pass.

    Returns:
        (loss_dict, cos_sim, coords_valid, loss_or_none)
        Returns (None, None, None, None) if no valid points.
    """
    fx, fy, cx, cy, H_orig, W_orig = intrinsics

    depth = data["depth"].to(device)
    dino_feat = data["dino_feat"].to(device)
    extrinsic = data["extrinsic"].to(device)
    robot_mask = data["robot_mask"].to(device)
    seg_inst = data["seg_inst"].to(device)
    state_idx = int(data["state_index"][0])

    # Unproject depth -> world coordinates
    cam_to_world = extrinsic.unsqueeze(1)
    coords_world = unproject_depth_to_world(
        depth, cam_to_world,
        fx=fx, fy=fy, cx=cx, cy=cy,
        original_height=H_orig, original_width=W_orig,
    )

    B, C, Hf, Wf = dino_feat.shape
    coords_flat = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)
    dino_flat = dino_feat.permute(0, 2, 3, 1).reshape(-1, C)
    mask_flat = robot_mask.reshape(-1)
    depth_flat = depth.reshape(-1)
    seg_flat = seg_inst.reshape(-1).long()

    # Filter: robot pixels with valid depth
    valid_mask = mask_flat & (depth_flat > 0.01)
    if valid_mask.sum() == 0:
        return None, None, None, None

    query_coords = coords_flat[valid_mask]
    target_dino = dino_flat[valid_mask]
    seg_ids_valid = seg_flat[valid_mask]

    # Get FK surface points for this state
    surface_world = surface_cache[state_idx]

    # Interpolate features + decode
    interp_feat = robot_points.interpolate_features(query_coords, surface_world)
    pred_dino = decoder(interp_feat)

    # 1. Reconstruction loss
    recon_loss, cos_sim = cosine_similarity_loss(pred_dino, target_dino)

    # 2. Part-level contrastive loss
    contrastive_loss = torch.tensor(0.0, device=device)
    lambda_contrastive = robot_losses["lambda_contrastive"]
    if lambda_contrastive > 0:
        Q = interp_feat.shape[0]
        n_unique = seg_ids_valid.unique().numel()
        if Q >= 2 and n_unique >= 2:
            M = min(Q, robot_losses["contrastive_max_samples"])
            indices = torch.randperm(Q, device=device)[:M]
            feat_sample = F.normalize(interp_feat[indices], dim=-1)
            seg_sample = seg_ids_valid[indices]

            positive_mask = seg_sample.unsqueeze(0) == seg_sample.unsqueeze(1)
            positive_mask.fill_diagonal_(False)
            valid_pair_mask = ~torch.eye(M, dtype=torch.bool, device=device)

            contrastive_loss = compute_infonce_loss(
                feat_sample, positive_mask, valid_pair_mask,
                robot_losses["contrastive_temperature"],
            )

    # Total loss
    loss = recon_loss + lambda_contrastive * contrastive_loss

    if not no_backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_dict = {
        "total": loss.detach(),
        "recon": recon_loss.detach(),
        "contrastive": (lambda_contrastive * contrastive_loss).detach(),
    }
    return loss_dict, cos_sim.mean().detach(), query_coords, loss if no_backward else None


# --------------------------------------------------------------------------- #
#  Training Loop
# --------------------------------------------------------------------------- #

def train(
    config: dict,
    device: str = "cuda",
    viser_server=None,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
) -> Tuple[RobotNeuralPointMap, MLP, RobotSurfaceSampler]:
    """Main training function.

    Args:
        config: Parsed YAML configuration dict.
        device: Torch device string.
        viser_server: Optional viser server for PCA visualization.
        use_wandb: Whether to use wandb logging.
        use_tensorboard: Whether to use tensorboard logging.

    Returns:
        (robot_points, decoder, sampler) — all on CPU.
    """
    cfg = RobotTrainConfig.from_dict(config)
    input_path = config["input_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # Dataset
    print(f"[Train] Loading dataset from {input_path}")
    dataset = RobotDataset(input_path, cfg.image_size, cfg.patch_size)
    print(f"[Train] {dataset.n_frames} frames, {dataset.n_states} states, "
          f"DINO dim={dataset.dino_dim}, feat_res={dataset.feat_h}x{dataset.feat_w}")

    intrinsics = (dataset.fx, dataset.fy, dataset.cx, dataset.cy,
                  dataset.H_orig, dataset.W_orig)

    batch_sampler = RobotBatchSampler(
        dataset.state_index, batch_size=cfg.batch_size, shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=robot_worker_init_fn,
    )

    # Surface sampler
    sampler_path = output_dir / "sampler.npz"
    if sampler_path.exists():
        sampler = RobotSurfaceSampler.load(sampler_path, URDF_PATH)
    else:
        sampler = RobotSurfaceSampler(URDF_PATH, voxel_size=cfg.voxel_size)
        sampler.save(sampler_path)

    # Pre-compute FK for all states
    surface_cache = precompute_surface_cache(
        sampler, dataset.urdf_cfgs, dataset.base_transforms, device,
    )

    # Models
    robot_points = RobotNeuralPointMap(
        n_points=sampler.n_total_points,
        feature_dim=cfg.feature_dim,
        knn_k=cfg.knn_k,
        temperature=cfg.temperature,
    ).to(device)

    decoder = MLP(
        input_dim=cfg.feature_dim,
        output_dim=dataset.dino_dim,
        hidden_dim=cfg.decoder_hidden_dim,
        num_res_blocks=cfg.decoder_num_res_blocks,
        dropout=cfg.decoder_dropout,
    ).to(device)

    # Load existing checkpoints if available
    points_path = output_dir / "robot_neural_points.pt"
    decoder_path = output_dir / "decoder.pt"
    if points_path.exists():
        robot_points.load_state_dict(
            torch.load(points_path, map_location=device, weights_only=True)
        )
        print(f"[LOAD] Loaded robot_neural_points from {points_path}")
    if decoder_path.exists():
        decoder.load_state_dict(
            torch.load(decoder_path, map_location=device, weights_only=True)
        )
        print(f"[LOAD] Loaded decoder from {decoder_path}")

    # Freeze decoder if not training
    if not cfg.train_decoder:
        decoder.requires_grad_(False)
        print("[Train] Decoder frozen (train_decoder=False)")

    # Optimizer
    params = list(robot_points.parameters())
    if cfg.train_decoder:
        params.extend(list(decoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    # Setup logging
    logger = TrainLogger(
        output_dir, Path(input_path).name, config,
        use_wandb=use_wandb, use_tensorboard=use_tensorboard,
    )

    # Import visualization if needed
    if cfg.run_pca and viser_server is not None:
        from utils.visualization import run_robot_pca_visualization

    # Build robot_losses dict from config
    robot_losses = {
        "lambda_contrastive": cfg.lambda_contrastive,
        "contrastive_temperature": cfg.contrastive_temperature,
        "contrastive_max_samples": cfg.contrastive_max_samples,
    }

    print(f"[Train] {sampler.n_total_points} surface points, "
          f"feature_dim={cfg.feature_dim}, knn_k={cfg.knn_k}")
    print(f"--- Starting training (epochs={cfg.num_epochs}, "
          f"batch_size={cfg.batch_size}, lr={cfg.learning_rate}) ---")

    # Training loop
    for epoch in range(cfg.num_epochs):
        loss_history = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}", file=sys.stdout)
        for i, data in enumerate(pbar):
            loss_dict, cos_sim_val, coords_valid, _ = train_step(
                data, robot_points, decoder, optimizer,
                surface_cache, intrinsics, robot_losses, device,
            )
            if loss_dict is None:
                continue

            loss_history.append(loss_dict["total"].detach())
            global_step = epoch * len(dataloader) + i

            # Logging (only call .item() at log_interval to avoid GPU sync every iteration)
            if (i + 1) % cfg.log_interval == 0:
                total_val = loss_dict["total"].item()
                loss_scalars = {k: v.item() for k, v in loss_dict.items()}
                logger.log_step(global_step, total_val, cos_sim_val.item(), coords_valid.shape[0])
                logger.log_contrastive_losses(global_step, loss_scalars)
                pbar.set_postfix_str(
                    f"Loss: {total_val:.4f} "
                    f"(R:{loss_scalars['recon']:.3f} Contr:{loss_scalars['contrastive']:.3f})"
                )

        # Epoch summary (sync GPU tensors once at epoch end)
        avg_loss = sum(l.item() for l in loss_history) / len(loss_history) if loss_history else 0
        print(f"[Epoch {epoch + 1}/{cfg.num_epochs}] Avg. Loss: {avg_loss:.4f}")

        logger.log_epoch(epoch + 1, avg_loss, sampler.n_total_points)

        # PCA Visualization (use canonical pose = state 0)
        if cfg.run_pca and viser_server is not None and cfg.vis_interval > 0:
            if (epoch + 1) % cfg.vis_interval == 0:
                run_robot_pca_visualization(
                    viser_server,
                    surface_cache[0],           # canonical pose surface points
                    robot_points.features,      # learned per-point features
                    decoder, "robot", epoch + 1, device, config,
                )

        # Save checkpoint
        if (epoch + 1) % cfg.save_interval == 0 or (epoch + 1) == cfg.num_epochs:
            torch.save(robot_points.state_dict(), points_path)
            if cfg.train_decoder:
                torch.save(decoder.state_dict(), decoder_path)
            print(f"[SAVE] Saved checkpoint to {output_dir}")

    # Cleanup
    logger.close()
    dataset.close()

    return robot_points.cpu(), decoder.cpu(), sampler


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train robot neural point map from hemisphere captures."
    )
    parser.add_argument(
        "--config", type=str, default="mapping/config/config_robot.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--input_path", type=str, default=None, help="Override input HDF5 path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable tensorboard logging")
    parser.add_argument("--no_viser", action="store_true", help="Disable viser visualization server")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.input_path is not None:
        config["input_path"] = args.input_path
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.num_epochs is not None:
        config["training"]["epochs"] = args.num_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Device: {device}")

    # Viser server
    viser_server = None
    if config.get("visualization", {}).get("enabled", False) and not args.no_viser:
        import viser
        viser_server = viser.ViserServer(host="0.0.0.0", port=8080)
        print("[Init] Viser server started at http://0.0.0.0:8080")

    use_wandb = config.get("wandb", {}).get("enabled", True) and not args.no_wandb
    use_tensorboard = not args.no_tensorboard

    robot_points, decoder, sampler = train(
        config, device=device,
        viser_server=viser_server,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )
    print(f"\n[Done] Training complete. Results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
