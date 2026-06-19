"""
Train latent feature decoder on point-based representation (NeuralPointMap) using HDF5 dataset.

This module provides the core training functionality that can be:
1. Run standalone via main()
2. Called from latent_mapping_task.py for distributed training
"""
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import h5py
# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.geometry import unproject_depth_to_world
from utils.optimizer import update_optimizer_param
from mapping.representations.neural_point_map import NeuralPointMap
from mapping.models.mlp import MLP
from mapping.data.env_dataset import EnvDataset, env_collate_fn, EnvBatchSampler, env_worker_init_fn
from mapping.losses.inter_category_loss import inter_category_loss, compute_cross_scene_inter_loss
from mapping.losses.cosine_similarity import cosine_similarity_loss
from mapping.losses.intra_instance_loss import intra_instance_loss

from mapping.config.env_train_config import EnvTrainConfig
from mapping.utils.logger import TrainLogger
from mapping.utils.point_map_io import load_or_create_point_maps, save_point_maps


# --------------------------------------------------------------------------- #
#  Dataset Setup
# --------------------------------------------------------------------------- #

def get_intrinsics_and_feature_dim(
    dataset_path: Path,
    target_envs: List[str],
    load_dino_features: bool,
    default_feature_dim: int,
) -> Tuple[float, float, float, float, int, int, int]:
    """
    Get camera intrinsics and feature dimension from first HDF5 file.

    Returns:
        (fx, fy, cx, cy, H_orig, W_orig, feature_dim)
    """
    # Find first HDF5 file
    first_hdf5 = None
    if target_envs:
        for env in target_envs:
            p = dataset_path / env
            if p.exists():
                if p.is_file():
                    first_hdf5 = p
                else:
                    first_hdf5 = next(p.glob("*.hdf5"), None)
                if first_hdf5:
                    break
    else:
        first_hdf5 = next(dataset_path.rglob("*.hdf5"), None)

    if not first_hdf5:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_path}")

    with h5py.File(first_hdf5, 'r') as f:
        if "intrinsics" not in f:
            raise ValueError("'intrinsics' not found in HDF5")
        K = f["intrinsics"][:]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H_orig, W_orig, _ = f['rgb'].shape[1:]

        if load_dino_features and 'dino' in f:
            feature_dim = f['dino'].shape[1]
        else:
            feature_dim = default_feature_dim

    return fx, fy, cx, cy, H_orig, W_orig, feature_dim


def create_dataset_and_loader(
    dataset_path: Path,
    cfg: EnvTrainConfig,
) -> Tuple[EnvDataset, DataLoader]:
    """Create EnvDataset and dataloader with EnvBatchSampler."""
    dataset = EnvDataset(
        dataset_path,
        cfg.target_envs,
        cfg.num_images,
        cfg.image_size,
        cfg.patch_size,
        load_dino_features=cfg.load_dino_features,
    )

    if len(dataset) == 0:
        raise ValueError("No valid training data found")

    batch_sampler = EnvBatchSampler(dataset, batch_size=cfg.batch_size, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=env_collate_fn,
        worker_init_fn=env_worker_init_fn,
    )

    return dataset, dataloader


# --------------------------------------------------------------------------- #
#  Training Step
# --------------------------------------------------------------------------- #

def filter_valid_points(
    coords_flat: torch.Tensor,
    depth_flat: torch.Tensor,
    feats_flat: torch.Tensor,
    inst_ids_flat: Optional[torch.Tensor],
    sam_ids_flat: Optional[torch.Tensor],
    excluded_ids: Optional[set],
    cfg: EnvTrainConfig,
    device: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Filter points by scene bounds, depth, and excluded instance IDs.

    Returns:
        (coords_valid, feats_valid, inst_ids_valid, sam_ids_valid) or (None, None, None, None) if no valid points
    """
    # Scene bounds filter
    in_x = (coords_flat[:, 0] > cfg.scene_min[0]) & (coords_flat[:, 0] < cfg.scene_max[0])
    in_y = (coords_flat[:, 1] > cfg.scene_min[1]) & (coords_flat[:, 1] < cfg.scene_max[1])
    in_z = (coords_flat[:, 2] > cfg.scene_min[2]) & (coords_flat[:, 2] < cfg.scene_max[2])
    valid_mask = in_x & in_y & in_z

    # Depth filter
    valid_mask = valid_mask & (depth_flat >= 0.01)

    if valid_mask.sum() == 0:
        return None, None, None, None

    coords_valid = coords_flat[valid_mask]
    feats_valid = feats_flat[valid_mask]
    inst_ids_valid = inst_ids_flat[valid_mask] if inst_ids_flat is not None else None
    sam_ids_valid = sam_ids_flat[valid_mask] if sam_ids_flat is not None else None

    # Excluded instance IDs filter (glass, mirror, etc.)
    if excluded_ids and len(excluded_ids) > 0 and inst_ids_valid is not None:
        excl_tensor = torch.tensor(list(excluded_ids), dtype=inst_ids_valid.dtype, device=device)
        keep_mask = ~torch.isin(inst_ids_valid, excl_tensor)

        if keep_mask.sum() == 0:
            return None, None, None, None

        coords_valid = coords_valid[keep_mask]
        feats_valid = feats_valid[keep_mask]
        inst_ids_valid = inst_ids_valid[keep_mask]
        if sam_ids_valid is not None:
            sam_ids_valid = sam_ids_valid[keep_mask]

    return coords_valid, feats_valid, inst_ids_valid, sam_ids_valid


def train_step(
    data: dict,
    point_maps: Dict[str, NeuralPointMap],
    decoder: MLP,
    encoder: MLP,
    decoder_optimizer: torch.optim.Optimizer,
    encoder_optimizer: Optional[torch.optim.Optimizer],
    cfg: EnvTrainConfig,
    intrinsics: Tuple[float, float, float, float, int, int],
    device: str,
    category_mapper=None,
    no_backward: bool = False,
    no_inter: bool = False,  # skip inter loss (when handled externally, e.g. joint training)
) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Execute a single training step.

    Decoder and point_map are trained on: recon + inter + intra losses.
    Encoder is trained on: consistency loss only (isolated, always runs).

    When no_backward=True the decoder backward/step is skipped and the raw
    decoder_loss tensor is returned as the 4th element so the caller can
    accumulate it with other losses before a single combined backward pass
    (used by train_neural_points_env_and_robot.py).

    Returns:
        (loss_dict, cos_sim_value, coords_valid, decoder_loss_or_none)
        decoder_loss_or_none is the live loss tensor when no_backward=True,
        else None (backward already done internally).
        Returns (None, None, None, None) if the step is skipped.
    """
    fx, fy, cx, cy, H_orig, W_orig = intrinsics

    # Get data tensors
    depth_t = data["depth_t"].to(device)
    extrinsic_t = data["extrinsic_t"].unsqueeze(1).to(device)
    env_name = data["env_name"][0]

    # Vision features
    if data.get("dino_feat") is None or data["dino_feat"][0] is None:
        return None, None, None, None
    vis_feat = data["dino_feat"].to(device)

    # 3D coordinates
    coords_world = unproject_depth_to_world(
        depth_t, extrinsic_t,
        fx=fx, fy=fy, cx=cx, cy=cy,
        original_height=H_orig, original_width=W_orig,
    )

    B, C_, Hf, Wf = vis_feat.shape

    # Flatten tensors
    feats_flat = vis_feat.permute(0, 2, 3, 1).reshape(-1, C_)
    coords_flat = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)
    depth_flat = depth_t.reshape(-1)

    # Instance IDs
    inst_ids_flat = None
    if data.get("seg_inst") is not None and data["seg_inst"][0] is not None:
        seg_inst = data["seg_inst"].to(device)
        inst_ids_flat = seg_inst.reshape(-1)

    # SAM mask IDs
    sam_ids_flat = None
    if data.get("sam_mask") is not None and data["sam_mask"][0] is not None:
        sam_mask = data["sam_mask"].to(device)
        sam_ids_flat = sam_mask.reshape(-1)

    # Excluded instance IDs
    excluded_ids = data.get("excluded_instance_ids")
    excluded_ids = excluded_ids[0] if excluded_ids else None

    # Instance to category mapping for contrastive loss
    inst_to_category = data.get("inst_to_category")
    inst_to_category = inst_to_category[0] if inst_to_category else {}

    # Filter valid points
    coords_valid, feats_valid, inst_ids_valid, sam_ids_valid = filter_valid_points(
        coords_flat, depth_flat, feats_flat, inst_ids_flat, sam_ids_flat,
        excluded_ids, cfg, device,
    )

    if coords_valid is None:
        return None, None, None, None

    # Get point map for this environment
    point_map = point_maps[env_name]

    # Initialize features for new points
    if cfg.use_encoder_init:
        # Use encoder output for initialization
        with torch.no_grad():
            latent_init = encoder(feats_valid).detach()
    else:
        # Random initialization (encoder ablation mode)
        latent_init = None

    # Register new points
    old_map_features = point_map.map_features
    num_added = point_map.register_points(
        coords_valid, new_features=latent_init,
        new_instance_ids=inst_ids_valid, skip_existing=True,
    )

    if num_added > 0:
        update_optimizer_param(decoder_optimizer, old_map_features, point_map.map_features)

    # Forward pass
    point_feat = point_map.interpolate_features(coords_valid)
    pred_feat = decoder(point_feat)

    # === Loss computation ===
    # 1. Reconstruction loss (cosine similarity)
    recon_loss, cos_sim = cosine_similarity_loss(pred_feat, feats_valid)

    # 2. Inter-category contrastive loss
    inter_loss = torch.tensor(0.0, device=device)
    if not no_inter and cfg.lambda_inter > 0 and category_mapper is not None:
        if cfg.cross_scene_inter:
            inter_loss = compute_cross_scene_inter_loss(
                point_maps,
                category_mapper,
                max_total_samples=cfg.inter_max_samples,
                temperature=cfg.inter_temperature,
                device=device,
            )
        elif inst_ids_valid is not None:
            inter_loss = inter_category_loss(
                point_feat,
                inst_ids_valid,
                inst_to_category,
                temperature=cfg.inter_temperature,
                max_samples=cfg.inter_max_samples,
                mapper=category_mapper,
            )

    # 3. Intra-instance contrastive loss (same SAM segment = positive, different segment = negative)
    intra_loss = torch.tensor(0.0, device=device)
    if cfg.lambda_intra > 0 and sam_ids_valid is not None:
        intra_loss = intra_instance_loss(
            point_feat,
            sam_ids_valid,
            instance_ids=inst_ids_valid,
            inst_to_category=inst_to_category,
            temperature=cfg.intra_temperature,
            max_samples=cfg.intra_max_samples,
            mapper=category_mapper,
        )

    decoder_loss = (
        recon_loss
        + cfg.lambda_inter * inter_loss
        + cfg.lambda_intra * intra_loss
    )

    # Encoder backward (always isolated — separate optimizer, no cross-contamination)
    consistency_loss = torch.tensor(0.0, device=device)
    if encoder_optimizer is not None and cfg.lambda_consistency > 0:
        encoded_feat = encoder(feats_valid)
        consistency_loss, _ = cosine_similarity_loss(encoded_feat, point_feat.detach())
        encoder_optimizer.zero_grad()
        (cfg.lambda_consistency * consistency_loss).backward()
        encoder_optimizer.step()

    # Decoder backward — deferred when no_backward=True (joint training)
    if not no_backward:
        decoder_optimizer.zero_grad()
        decoder_loss.backward()
        decoder_optimizer.step()

    # Total loss for logging
    loss = decoder_loss.detach() + cfg.lambda_consistency * consistency_loss.detach()

    loss_dict = {
        "total": loss.detach(),
        "recon": recon_loss.detach(),
        "inter": (cfg.lambda_inter * inter_loss).detach(),
        "intra": (cfg.lambda_intra * intra_loss).detach(),
        "consistency": (cfg.lambda_consistency * consistency_loss).detach(),
    }

    return (
        loss_dict,
        cos_sim.mean().detach(),
        coords_valid,
        decoder_loss if no_backward else None,
    )


# --------------------------------------------------------------------------- #
#  Core Training Function
# --------------------------------------------------------------------------- #

def train(
    dataset_dir: str,
    output_dir: str,
    config: dict,
    decoder: MLP = None,
    encoder: MLP = None,
    device: str = "cuda",
    viser_server=None,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
) -> Tuple[Dict[str, NeuralPointMap], MLP, MLP]:
    """
    Main training function for NeuralPointMap.

    Args:
        dataset_dir: Path to directory containing HDF5 files
        output_dir: Path to save neural_points and decoder
        config: Training configuration dict
        decoder: Optional pre-loaded decoder (for federated learning)
        encoder: Optional pre-loaded encoder (for federated learning)
        device: torch device string
        viser_server: Optional viser server for PCA visualization
        use_wandb: Whether to use wandb logging
        use_tensorboard: Whether to use tensorboard logging

    Returns:
        point_maps: dict of env_name -> NeuralPointMap (on CPU)
        decoder: Trained MLP (latent → DINO)
        encoder: Trained MLP (DINO → latent)
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse configuration
    cfg = EnvTrainConfig.from_dict(config)

    # Get intrinsics and feature dimension
    model_config = config['dino_model'] if cfg.model_type == "dino" else config['clip_model']
    default_feature_dim = model_config.get('feature_dim', 1280)

    fx, fy, cx, cy, H_orig, W_orig, feature_dim = get_intrinsics_and_feature_dim(
        dataset_path, cfg.target_envs, cfg.load_dino_features, default_feature_dim
    )
    intrinsics = (fx, fy, cx, cy, H_orig, W_orig)

    print(f"[Train] Feature dimension: {feature_dim}")
    print(f"[Train] Dataset: {dataset_path}")
    print(f"[Train] Output: {output_path}")
    print(f"[Train] Losses: recon×1.0, inter×{cfg.lambda_inter}, intra×{cfg.lambda_intra}, consistency×{cfg.lambda_consistency}")

    # Create dataset and dataloader
    dataset, dataloader = create_dataset_and_loader(dataset_path, cfg)
    print(f"[Train] Loaded {len(dataset)} samples from {len(dataset.env_names)} environments")

    # Save shared instance_id_to_name mapping
    if dataset.instance_id_to_name:
        id_map_path = output_path / "instance_id_to_name.json"
        with open(id_map_path, 'w') as f:
            json.dump(dataset.instance_id_to_name, f, indent=2)
        print(f"[Train] Saved instance_id_to_name to {id_map_path}")

    # Initialize or use provided decoder
    if decoder is None:
        decoder = MLP(
            input_dim=cfg.feature_dim,
            output_dim=feature_dim,
            hidden_dim=cfg.decoder_hidden_dim,
            num_res_blocks=cfg.decoder_num_res_blocks,
            dropout=cfg.decoder_dropout,
        ).to(device)

        decoder_path = output_path / "decoder.pt"
        if decoder_path.exists():
            decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
            print(f"[LOAD] Loaded existing decoder from {decoder_path}")

    # Initialize or use provided encoder
    if encoder is None:
        encoder = MLP(
            input_dim=feature_dim,  # DINO feature dimension
            output_dim=cfg.feature_dim,  # Latent feature dimension
            hidden_dim=cfg.encoder_hidden_dim,
            num_res_blocks=cfg.encoder_num_res_blocks,
            dropout=cfg.encoder_dropout,
        ).to(device)

        encoder_path = output_path / "encoder.pt"
        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
            print(f"[LOAD] Loaded existing encoder from {encoder_path}")

    # Freeze decoder/encoder if not training
    if not cfg.train_decoder:
        decoder.requires_grad_(False)
        print("[Train] Decoder frozen (train_decoder=False)")
    if not cfg.train_encoder:
        encoder.requires_grad_(False)
        print("[Train] Encoder frozen (train_encoder=False)")

    # Load or create point maps
    split = config.get('split', 'train')
    point_maps = load_or_create_point_maps(
        dataset.env_names, cfg, output_path, device,
        split=split, require_existing=not cfg.train_decoder and not cfg.use_encoder_init,
    )

    # Setup separate optimizers for gradient isolation
    decoder_params = []
    if cfg.train_decoder:
        decoder_params.extend(list(decoder.parameters()))
    for point_map in point_maps.values():
        decoder_params.extend(list(point_map.parameters()))
    decoder_optimizer = torch.optim.Adam(decoder_params, lr=cfg.learning_rate)

    encoder_optimizer = None
    if cfg.train_encoder:
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.learning_rate)

    # Setup logging
    logger = TrainLogger(
        output_path, dataset_path.name, config,
        use_wandb=use_wandb, use_tensorboard=use_tensorboard
    )

    # Import visualization if needed
    if cfg.run_pca and viser_server is not None:
        from utils.visualization import run_pca_visualization

    # Pre-build CategoryMapper once (invariant across all training steps)
    category_mapper = None
    if dataset.instance_to_category:
        from mapping.losses.inter_category_loss import CategoryMapper
        category_mapper = CategoryMapper(dataset.instance_to_category, device=device)

    print(f"--- Starting training ({len(dataset)} samples, batch_size={cfg.batch_size}) ---")

    # Training loop
    for epoch in range(cfg.num_epochs):
        loss_history = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", file=sys.stdout)
        for i, data in enumerate(pbar):
            if not data:
                continue

            loss_dict, cos_sim_val, coords_valid, _ = train_step(
                data, point_maps, decoder, encoder,
                decoder_optimizer, encoder_optimizer,
                cfg, intrinsics, device,
                category_mapper=category_mapper,
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
                    f"(R:{loss_scalars['recon']:.3f} Inter:{loss_scalars['inter']:.3f} "
                    f"Intra:{loss_scalars['intra']:.3f} Cons:{loss_scalars['consistency']:.3f})"
                )

        # Epoch summary (sync GPU tensors once at epoch end)
        avg_loss = sum(l.item() for l in loss_history) / len(loss_history) if loss_history else 0
        total_points = sum(pm.map_points.shape[0] for pm in point_maps.values())
        print(f"[Epoch {epoch+1}/{cfg.num_epochs}] Avg. Loss: {avg_loss:.4f}")

        logger.log_epoch(epoch + 1, avg_loss, total_points)

        # PCA Visualization (only for first environment)
        if cfg.run_pca and viser_server is not None and cfg.vis_interval > 0:
            if (epoch + 1) % cfg.vis_interval == 0:
                first_env_name = next(iter(point_maps.keys()))
                filtered_ids = set()

                # Convert category names to instance IDs if include_categories is set
                include_ids = None
                vis_config = config.get('visualization', {})
                include_categories = vis_config.get('include_categories', [])
                if include_categories:
                    inst_to_cat = dataset.instance_to_category
                    include_ids = {iid for iid, cat in inst_to_cat.items() if cat in include_categories}
                    if include_ids:
                        print(f"[VIS] Filtering PCA to categories {include_categories}: {len(include_ids)} instance IDs")

                run_pca_visualization(
                    viser_server, point_maps[first_env_name], decoder,
                    first_env_name, epoch + 1, device, config, filtered_ids,
                    include_instance_ids=include_ids
                )

        # Save checkpoint
        is_last_epoch = (epoch + 1) == cfg.num_epochs
        if (epoch + 1) % cfg.save_interval == 0 or is_last_epoch:
            save_point_maps(point_maps, output_path, split=split)

            if cfg.train_decoder:
                decoder_path = output_path / "decoder.pt"
                torch.save(decoder.state_dict(), decoder_path)
                print(f"[SAVE] Saved decoder to {decoder_path}")

            if cfg.train_encoder:
                encoder_path = output_path / "encoder.pt"
                torch.save(encoder.state_dict(), encoder_path)
                print(f"[SAVE] Saved encoder to {encoder_path}")

    # Cleanup
    logger.close()

    # Move point maps to CPU before returning
    for env_name in point_maps:
        point_maps[env_name] = point_maps[env_name].cpu()

    return point_maps, decoder, encoder


# --------------------------------------------------------------------------- #
#  Main (standalone execution)
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train NeuralPointMap with latent features.")
    parser.add_argument("--config", type=str, default="mapping/config/config_env.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Override dataset directory from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="Disable tensorboard logging")
    parser.add_argument("--no_viser", action="store_true",
                        help="Disable viser visualization server")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine paths
    dataset_dir = args.dataset_dir or config['dataset_dir']
    output_dir = args.output_dir or config.get('output_dir', 'map_output')

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Device: {device}")

    # Viser server
    viser_server = None
    if config.get('visualization', {}).get('enabled', False) and not args.no_viser:
        import viser
        viser_server = viser.ViserServer(host="0.0.0.0", port=8080)
        print("[Init] Viser server started at http://0.0.0.0:8080")

    # Save config to output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.yaml", 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # Run training
    use_wandb = config.get('wandb', {}).get('enabled', True) and not args.no_wandb
    use_tensorboard = not args.no_tensorboard

    point_maps, decoder, encoder = train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=config,
        decoder=None,
        device=device,
        viser_server=viser_server,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )

    print(f"\n[Done] Training complete. Results saved to {output_dir}")

    # Keep viser running if enabled
    if viser_server is not None and config.get('visualization', {}).get('enabled', False):
        print("[VIS] Viser server running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
