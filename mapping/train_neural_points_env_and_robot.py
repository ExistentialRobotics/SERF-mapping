"""Joint training of env + robot neural point maps with a shared decoder.

Combines train_neural_points_env.py and train_neural_points_robot.py into a
single training loop.  A shared decoder and inter-category feature bank ensure
that robot features are semantically distinguished from all env categories in
the same latent space.

Usage:
    python mapping/train_neural_points_env_and_robot.py \
        --config mapping/config/config_env_and_robot.yaml \
        --no_wandb --no_viser
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.robot import URDF_PATH, RobotSurfaceSampler
from utils.visualization import run_joint_pca_visualization
from mapping.representations.neural_point_map import NeuralPointMap
from mapping.representations.robot_neural_points import RobotNeuralPointMap
from mapping.models.mlp import MLP
from mapping.data.robot_dataset import RobotDataset, RobotBatchSampler, robot_worker_init_fn
from mapping.losses.inter_category_loss import CategoryMapper, compute_cross_scene_inter_loss
from mapping.config.env_train_config import EnvTrainConfig
from mapping.utils.logger import TrainLogger
from mapping.utils.point_map_io import load_or_create_point_maps, save_point_maps

from mapping.train_neural_points_env import (
    get_intrinsics_and_feature_dim,
    create_dataset_and_loader,
    train_step as env_train_step,
)
from mapping.train_neural_points_robot import (
    precompute_surface_cache,
    train_step as robot_train_step,
)


# --------------------------------------------------------------------------- #
#  Main Training Function
# --------------------------------------------------------------------------- #

def train(
    config: dict,
    device: str = "cuda",
    decoder: MLP | None = None,
    encoder: MLP | None = None,
    viser_server=None,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    abort_event: threading.Event | None = None,
) -> Tuple[Dict[str, NeuralPointMap], RobotNeuralPointMap, MLP, MLP]:
    """Joint training of env + robot neural point maps with shared decoder.

    Args:
        config: Parsed YAML configuration dict.
        device: Torch device string.
        decoder: Optional pre-loaded decoder (for federated learning).
        encoder: Optional pre-loaded encoder (for federated learning).
        viser_server: Optional viser server for PCA visualization.
        use_wandb: Whether to use wandb logging.
        use_tensorboard: Whether to use tensorboard logging.

    Returns:
        (env_point_maps, robot_points, decoder, encoder) — all on CPU.
    """
    # --- Parse configs ---
    env_cfg = EnvTrainConfig.from_dict(config)

    # Robot loss hyperparameters — reuse intra loss settings for consistency
    robot_losses = {
        "lambda_contrastive": env_cfg.lambda_intra,
        "contrastive_temperature": env_cfg.intra_temperature,
        "contrastive_max_samples": env_cfg.intra_max_samples,
    }

    dataset_path = Path(config["dataset_dir"])
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    robot_input_path = config["robot_input_path"]
    robot_output_dir = output_path / "neural_points" / "robot"
    robot_output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # --- Env dataset + intrinsics ---
    model_config = config["dino_model"] if env_cfg.model_type == "dino" else config["clip_model"]
    default_feature_dim = model_config.get("feature_dim", 1280)

    fx, fy, cx, cy, H_orig, W_orig, feature_dim = get_intrinsics_and_feature_dim(
        dataset_path, env_cfg.target_envs, env_cfg.load_dino_features, default_feature_dim,
    )
    env_intrinsics = (fx, fy, cx, cy, H_orig, W_orig)

    print(f"[Train] Feature dimension: {feature_dim}")
    print(f"[Train] Env dataset: {dataset_path}")
    print(f"[Train] Robot input: {robot_input_path}")
    print(f"[Train] Output: {output_path}")
    print(f"[Train] Losses: recon×1.0, inter×{env_cfg.lambda_inter}, intra×{env_cfg.lambda_intra}, consistency×{env_cfg.lambda_consistency}")

    env_dataset, env_dataloader = create_dataset_and_loader(dataset_path, env_cfg)
    print(f"[Train] Env: {len(env_dataset)} samples from {len(env_dataset.env_names)} environments")

    # Save shared instance_id_to_name
    if env_dataset.instance_id_to_name:
        id_map_path = output_path / "instance_id_to_name.json"
        with open(id_map_path, "w") as f:
            json.dump(env_dataset.instance_id_to_name, f, indent=2)

    # --- Robot dataset ---
    robot_dataset = RobotDataset(
        robot_input_path, env_cfg.image_size, env_cfg.patch_size,
    )
    robot_intrinsics = (
        robot_dataset.fx, robot_dataset.fy, robot_dataset.cx, robot_dataset.cy,
        robot_dataset.H_orig, robot_dataset.W_orig,
    )

    robot_batch_sampler = RobotBatchSampler(
        robot_dataset.state_index, batch_size=env_cfg.batch_size, shuffle=True,
    )
    robot_dataloader = DataLoader(
        robot_dataset,
        batch_sampler=robot_batch_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=robot_worker_init_fn,
    )
    print(f"[Train] Robot: {robot_dataset.n_frames} frames, {robot_dataset.n_states} states, "
          f"DINO dim={robot_dataset.dino_dim}")

    # --- Robot surface sampler + FK cache ---
    sampler_path = robot_output_dir / "sampler.npz"
    if sampler_path.exists():
        sampler = RobotSurfaceSampler.load(sampler_path, URDF_PATH)
    else:
        sampler = RobotSurfaceSampler(URDF_PATH, voxel_size=config["point_map"]["voxel_size"])
        sampler.save(sampler_path)

    surface_cache = precompute_surface_cache(
        sampler, robot_dataset.urdf_cfgs, robot_dataset.base_transforms, device,
    )

    # --- Models (shared decoder) ---
    decoder_path = output_path / "decoder.pt"
    encoder_path = output_path / "encoder.pt"

    if decoder is None:
        decoder = MLP(
            input_dim=env_cfg.feature_dim,
            output_dim=feature_dim,
            hidden_dim=env_cfg.decoder_hidden_dim,
            num_res_blocks=env_cfg.decoder_num_res_blocks,
            dropout=env_cfg.decoder_dropout,
        ).to(device)

        if decoder_path.exists():
            decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
            print(f"[LOAD] Loaded existing decoder from {decoder_path}")

    if encoder is None:
        encoder = MLP(
            input_dim=feature_dim,
            output_dim=env_cfg.feature_dim,
            hidden_dim=env_cfg.encoder_hidden_dim,
            num_res_blocks=env_cfg.encoder_num_res_blocks,
            dropout=env_cfg.encoder_dropout,
        ).to(device)

        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
            print(f"[LOAD] Loaded existing encoder from {encoder_path}")

    if not env_cfg.train_decoder:
        decoder.requires_grad_(False)
        print("[Train] Decoder frozen (train_decoder=False)")
    if not env_cfg.train_encoder:
        encoder.requires_grad_(False)
        print("[Train] Encoder frozen (train_encoder=False)")

    # --- Point maps ---
    split = config.get("split", "train")
    episode_splits = config.get("episode_splits", {})
    env_point_maps = load_or_create_point_maps(
        env_dataset.env_names, env_cfg, output_path, device,
        split=split, episode_splits=episode_splits,
        require_existing=not env_cfg.train_decoder and not env_cfg.use_encoder_init,
    )

    robot_points = RobotNeuralPointMap(
        n_points=sampler.n_total_points,
        feature_dim=env_cfg.feature_dim,
        knn_k=env_cfg.knn_k,
        temperature=config["point_map"].get("temperature", 0.05),
    ).to(device)

    robot_points_path = robot_output_dir / "robot_neural_points.pt"
    if robot_points_path.exists():
        robot_points.load_state_dict(
            torch.load(robot_points_path, map_location=device, weights_only=True)
        )
        print(f"[LOAD] Loaded robot_neural_points from {robot_points_path}")

    # --- Optimizers ---
    # Single optimizer for decoder + env point maps + robot point features
    decoder_params = []
    if env_cfg.train_decoder:
        decoder_params.extend(list(decoder.parameters()))
    for pm in env_point_maps.values():
        decoder_params.extend(list(pm.parameters()))
    decoder_params.extend(list(robot_points.parameters()))
    decoder_optimizer = torch.optim.Adam(decoder_params, lr=env_cfg.learning_rate)

    encoder_optimizer = None
    if env_cfg.train_encoder:
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=env_cfg.learning_rate)

    # --- Category mapper ---
    category_mapper = None
    if env_dataset.instance_to_category:
        category_mapper = CategoryMapper(env_dataset.instance_to_category, device=device)

    # --- Logging ---
    logger = TrainLogger(
        output_path, dataset_path.name, config,
        use_wandb=use_wandb, use_tensorboard=use_tensorboard,
    )

    print(f"[Train] Robot surface points: {sampler.n_total_points}, "
          f"feature_dim={env_cfg.feature_dim}, knn_k={env_cfg.knn_k}")
    print(f"--- Starting joint training (env: {len(env_dataset)} samples, "
          f"robot: {robot_dataset.n_frames} frames) ---")

    # --- Training loop ---
    num_epochs = env_cfg.num_epochs
    for epoch in range(num_epochs):
        # --- Watchdog: check for abort ---
        if abort_event is not None and abort_event.is_set():
            print(f"[Train] Abort signal received at epoch {epoch + 1}. Exiting early.")
            break

        env_loss_history = []
        robot_loss_history = []

        # Create iterators
        env_iter = iter(env_dataloader)
        robot_iter = iter(robot_dataloader)
        max_steps = max(len(env_dataloader), len(robot_dataloader))

        pbar = tqdm(range(max_steps), desc=f"Epoch {epoch + 1}/{num_epochs}", file=sys.stdout)
        for step in pbar:
            global_step = epoch * max_steps + step

            # --- Env train step (no_backward=True: defer decoder backward) ---
            env_data = next(env_iter, None)
            if env_data is None:
                env_iter = iter(env_dataloader)
                env_data = next(env_iter, None)

            env_loss_dict, env_cos_sim, _, env_decoder_loss = env_train_step(
                env_data, env_point_maps, decoder, encoder,
                decoder_optimizer, encoder_optimizer,
                env_cfg, env_intrinsics, device,
                category_mapper=category_mapper,
                no_backward=True,
                no_inter=env_cfg.cross_scene_inter,  # joint training handles cross-scene inter
            ) if env_data else (None, None, None, None)
            if env_loss_dict is not None:
                env_loss_history.append(env_loss_dict["total"].detach())

            # --- Robot train step (no_backward=True: defer decoder backward) ---
            robot_data = next(robot_iter, None)
            if robot_data is None:
                robot_iter = iter(robot_dataloader)
                robot_data = next(robot_iter, None)

            robot_loss_dict, robot_cos_sim, _, robot_decoder_loss = robot_train_step(
                robot_data, robot_points, decoder, decoder_optimizer,
                surface_cache, robot_intrinsics, robot_losses, device,
                no_backward=True,
            ) if robot_data is not None else (None, None, None, None)
            if robot_loss_dict is not None:
                robot_loss_history.append(robot_loss_dict["total"].detach())

            # --- Joint cross-scene inter-category loss (env all scenes + robot) ---
            joint_inter_loss = torch.tensor(0.0, device=device)
            if env_cfg.lambda_inter > 0 and env_cfg.cross_scene_inter and category_mapper is not None:
                robot_extra = None
                if robot_data is not None:
                    state_idx = int(robot_data["state_index"][0])
                    robot_surf = surface_cache[state_idx]                                    # (N_surface, 3)
                    robot_interp = robot_points.interpolate_features(robot_surf, robot_surf) # (N_surface, D)
                    n_robot = max(1, min(
                        env_cfg.inter_max_samples // (category_mapper.num_categories + 1),
                        robot_interp.shape[0],
                    ))
                    idx = torch.randperm(robot_interp.shape[0], device=device)[:n_robot]
                    robot_extra = robot_interp[idx]                                          # (n_per_cat, D)

                joint_inter_loss = compute_cross_scene_inter_loss(
                    env_point_maps,
                    category_mapper,
                    max_total_samples=env_cfg.inter_max_samples,
                    temperature=env_cfg.inter_temperature,
                    device=device,
                    extra_features=robot_extra,
                    extra_category_id=category_mapper.num_categories,
                )

            # --- Combined decoder backward (env + robot + joint_inter, one pass) ---
            losses = [l for l in [env_decoder_loss, robot_decoder_loss] if l is not None]
            if losses:
                combined = sum(losses)
                if env_cfg.cross_scene_inter:
                    combined = combined + env_cfg.lambda_inter * joint_inter_loss
                decoder_optimizer.zero_grad()
                combined.backward()
                decoder_optimizer.step()

            # --- Logging ---
            if (step + 1) % env_cfg.log_interval == 0:
                parts = []
                joint_inter_val = (env_cfg.lambda_inter * joint_inter_loss).detach().item() if env_cfg.cross_scene_inter else 0.0

                if env_loss_dict is not None:
                    ev = {k: v.item() for k, v in env_loss_dict.items()}
                    i_val = joint_inter_val if env_cfg.cross_scene_inter else ev["inter"]
                    parts.append(
                        f"Env: {ev['total']:.4f} "
                        f"(R:{ev['recon']:.3f} I:{i_val:.3f} A:{ev['intra']:.3f} Cons:{ev['consistency']:.3f})"
                    )
                    logger.log_step(global_step, ev["total"], env_cos_sim.item() if env_cos_sim is not None else 0, 0)
                    logger.log_contrastive_losses(global_step, {f"env/{k}": v for k, v in ev.items()})

                if robot_loss_dict is not None:
                    rv = {k: v.item() for k, v in robot_loss_dict.items()}
                    parts.append(
                        f"Robot: {rv['total']:.4f} "
                        f"(R:{rv['recon']:.3f} C:{rv['contrastive']:.3f})"
                    )
                    logger.log_contrastive_losses(global_step, {f"robot/{k}": v for k, v in rv.items()})

                if env_cfg.cross_scene_inter:
                    logger.log_contrastive_losses(global_step, {"joint/inter": joint_inter_val})

                if parts:
                    pbar.set_postfix_str("  |  ".join(parts))

        # --- Epoch summary ---
        avg_env = sum(l.item() for l in env_loss_history) / len(env_loss_history) if env_loss_history else 0
        avg_robot = sum(l.item() for l in robot_loss_history) / len(robot_loss_history) if robot_loss_history else 0
        total_env_pts = sum(pm.map_points.shape[0] for pm in env_point_maps.values())
        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Env Loss: {avg_env:.4f} ({total_env_pts} pts)  "
              f"Robot Loss: {avg_robot:.4f} ({sampler.n_total_points} pts)")

        logger.log_epoch(epoch + 1, avg_env, total_env_pts)

        # --- Joint PCA visualization ---
        if env_cfg.run_pca and viser_server is not None and env_cfg.vis_interval > 0:
            if (epoch + 1) % env_cfg.vis_interval == 0:
                run_joint_pca_visualization(
                    viser_server, env_point_maps, robot_points,
                    surface_cache[0], decoder, epoch + 1, device, config,
                )

        # --- Save checkpoints ---
        is_last = (epoch + 1) == num_epochs
        if (epoch + 1) % env_cfg.save_interval == 0 or is_last:
            # Env point maps
            save_point_maps(env_point_maps, output_path, split=split,
                           episode_splits=episode_splits)

            # Robot point map
            torch.save(robot_points.state_dict(), robot_points_path)
            print(f"[SAVE] Saved robot_neural_points to {robot_points_path}")

            # Shared decoder
            if env_cfg.train_decoder:
                torch.save(decoder.state_dict(), decoder_path)
                print(f"[SAVE] Saved shared decoder to {decoder_path}")

            # Encoder
            if env_cfg.train_encoder:
                torch.save(encoder.state_dict(), encoder_path)
                print(f"[SAVE] Saved encoder to {encoder_path}")

        # --- Report progress to watchdog ---
        if progress_callback is not None:
            progress_callback(epoch + 1, num_epochs)

    # --- Cleanup ---
    logger.close()
    robot_dataset.close()

    for env_name in env_point_maps:
        env_point_maps[env_name] = env_point_maps[env_name].cpu()

    return env_point_maps, robot_points.cpu(), decoder.cpu(), encoder.cpu()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint env + robot neural point map training with shared decoder."
    )
    parser.add_argument(
        "--config", type=str, default="mapping/config/config_env_and_robot.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--dataset_dir", type=str, default=None, help="Override env dataset directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable tensorboard logging")
    parser.add_argument("--no_viser", action="store_true", help="Disable viser visualization server")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.dataset_dir is not None:
        config["dataset_dir"] = args.dataset_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

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

    env_point_maps, robot_points, decoder, encoder = train(
        config, device=device,
        viser_server=viser_server,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )
    print(f"\n[Done] Joint training complete. Results saved to {config['output_dir']}")

    # Keep viser running if enabled
    if viser_server is not None:
        print("[VIS] Viser server running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
