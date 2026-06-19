"""Iterative keypoint extraction, tracking, and neural point map updating (multiview)."""

import argparse
import os
import sys
from collections import deque

# Replace script directory (tracking/) with project root so that top-level
# packages resolve consistently when this file is executed as a script.
sys.path[0] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

from mapping.representations.neural_point_map import NeuralPointMap
from mapping.utils.point_map_io import find_neural_point_file, load_point_map_from_state_dict
from tracking.config.tracking_config import TrackingConfig, load_task_config
from tracking.utils import (
    count_registration_correspondences_per_instance,
    export_tracking_results,
    extract_keypoints_automatically,
    get_active_foreground_indices,
    register_instance_for_frame,
    render_2d_frame,
    render_3d_frame,
    run_cotracker_offline,
    sample_random_mask_points,
    select_fallback_retry_instances_and_views,
)
from utils.instance_ids import (
    build_cross_episode_id_mapping,
    filter_ids_by_keywords,
    load_instance_id_map_from_hdf5,
    load_instance_id_map_from_json,
    resolve_names_to_ids,
)
from utils.visualization import apply_camera_extrinsic, compute_pca_colors


# --- Helper: Load Neural Point Map --- #

def _load_neural_point_map(cfg: TrackingConfig, device: str) -> NeuralPointMap:
    """Load a pre-trained neural point map from the run directory.

    Searches for the .pt file matching the episode name in ``cfg.data_path``
    under ``cfg.run_dir/neural_points/``.
    """
    model_config_path = os.path.join(cfg.run_dir, "config.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    episode_name = os.path.splitext(os.path.basename(cfg.data_path))[0]
    particles_path = find_neural_point_file(cfg.run_dir, episode_name)

    state_dict = torch.load(particles_path, map_location="cpu")
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
    return neural_points


def _resolve_tracking_video_path(path_template: str, episode_name: str) -> str:
    """Render a video path template and keep the output under tracking/."""
    try:
        rendered_path = path_template.format(episode_name=episode_name)
    except (KeyError, ValueError) as exc:
        raise ValueError("Video output paths only support the {episode_name} template field.") from exc

    default_video_dir = os.path.join("tracking", "videos")
    if os.path.isabs(rendered_path):
        rendered_path = os.path.join(default_video_dir, os.path.basename(rendered_path))
    else:
        rendered_path = os.path.normpath(rendered_path)
        if rendered_path.split(os.sep)[0] != "tracking":
            rendered_path = os.path.join(default_video_dir, os.path.basename(rendered_path))

    os.makedirs(os.path.dirname(rendered_path), exist_ok=True)
    return rendered_path


# --- Helper: Instance ID Context --- #

def _build_instance_id_context(
    cfg: TrackingConfig,
    hdf5_file: h5py.File,
    neural_points: NeuralPointMap,
    device: str,
) -> tuple[dict, dict, torch.Tensor, list[int]]:
    """Build instance ID mappings, exclusion sets, and target IDs.

    Returns:
        online_to_train_id: Mapping from online instance IDs to training IDs.
        training_id_map: Training instance ID to name mapping.
        excluded_train_ids_tensor: Tensor of excluded background/structural IDs.
        target_instance_ids: List of online instance IDs to track.
    """
    train_json_path = os.path.join(cfg.run_dir, "instance_id_to_name.json")
    training_id_map = {}
    if os.path.exists(train_json_path):
        print(f"[INFO] Loading training instance map from: {train_json_path}")
        training_id_map = load_instance_id_map_from_json(train_json_path)
    else:
        print(f"[WARN] Training instance_id_to_name.json not found in {cfg.run_dir}.")

    online_id_map = load_instance_id_map_from_hdf5(hdf5_file)
    online_to_train_id = build_cross_episode_id_mapping(online_id_map, training_id_map)
    print(f"[INFO] Built ID mapping for {len(online_to_train_id)} instances.")

    excluded_train_ids = filter_ids_by_keywords(training_id_map, cfg.exclude_categories)
    excluded_train_ids_tensor = torch.tensor(sorted(excluded_train_ids), dtype=torch.long, device=device)
    print(f"[INFO] Excluding {len(excluded_train_ids)} background/structural instances from visualization.")

    target_instance_ids = resolve_names_to_ids(online_id_map, cfg.target_instance_names)
    print(f"[ID RESOLVE] target_instance_ids = {target_instance_ids}")

    neural_points.refine_instance_ids_by_graph(
        distance=cfg.graph_refine_distance, majority_ratio=cfg.graph_refine_majority_ratio,
    )

    return online_to_train_id, training_id_map, excluded_train_ids_tensor, target_instance_ids


# --- Online Streaming Loop --- #

def _run_streaming_loop(
    cfg: TrackingConfig,
    start_frame: int,
    end_frame: int,
    views: list[str],
    hdf5_file: h5py.File,
    instance_particles: dict[int, torch.Tensor],
    intrinsics: dict[str, tuple[float, float, float, float]],
    cotracker: torch.nn.Module,
    neural_points: NeuralPointMap,
    online_to_train_id: dict,
    target_instance_ids: list[int],
    excluded_train_ids_tensor: torch.Tensor,
    committed_cumulative: dict[int, np.ndarray],
    instance_centroids: dict[int, np.ndarray],
    frame_indices: list[int],
    committed_frame_transforms: dict[int, list[np.ndarray]],
    moved_instances: dict[int, int],
    device: str,
    pca_colors_arr: np.ndarray | None = None,
    vis=None,
    pcd=None,
    writer=None,
    writer_2d=None,
) -> np.ndarray | None:
    """Process frames one at a time using a sliding window buffer.

    Instead of processing fixed-size chunks, maintains a rolling buffer of
    ``cfg.buffer_size`` frames per view.  For each new frame the buffer is
    shifted (oldest frame dropped, new frame appended) and CoTracker is run
    on the full buffer.  Only the *last* frame in the buffer is registered
    against the second-to-last frame (anchor).

    Returns:
        Updated ``pca_colors_arr`` (or None when video is disabled).
    """
    buffer_size = cfg.buffer_size

    # --- Per-view buffers (deque for O(1) append/drop) --- #
    buffers: dict[str, dict[str, deque]] = {}
    for view in views:
        grp = hdf5_file[view]
        first_rgb = grp["rgb"][start_frame]
        first_depth = grp["depth"][start_frame].astype(np.float32) / 1000.0
        first_pose = grp["poses"][start_frame].astype(np.float32)
        first_seg = grp["seg_instance_id"][start_frame]

        buffers[view] = {
            "rgb": deque([first_rgb] * buffer_size, maxlen=buffer_size),
            "depth": deque([first_depth] * buffer_size, maxlen=buffer_size),
            "poses": deque([first_pose] * buffer_size, maxlen=buffer_size),
            "seg": deque([first_seg] * buffer_size, maxlen=buffer_size),
        }

    frame_step = cfg.frame_step
    for iter_idx, frame_idx in enumerate(tqdm(range(start_frame, end_frame, frame_step), desc="Streaming")):
        # --- Shift buffer (skip for the very first frame) --- #
        if frame_idx > start_frame:
            for view in views:
                grp = hdf5_file[view]
                new_rgb = grp["rgb"][frame_idx]
                new_depth = grp["depth"][frame_idx].astype(np.float32) / 1000.0
                new_pose = grp["poses"][frame_idx].astype(np.float32)
                new_seg = grp["seg_instance_id"][frame_idx]

                buf = buffers[view]
                buf["rgb"].append(new_rgb)
                buf["depth"].append(new_depth)
                buf["poses"].append(new_pose)
                buf["seg"].append(new_seg)

        # --- Dynamic instance initialization --- #
        untracked_ids = [iid for iid in target_instance_ids if iid not in instance_particles]
        if untracked_ids:
            active_mask = (
                (neural_points.buffer_pt_index != neural_points.EMPTY_KEY)
                & (neural_points.buffer_pt_index != neural_points.DELETED_KEY)
            )
            active_idx = neural_points.buffer_pt_index[active_mask]
            active_inst_ids = neural_points.map_instance_ids[active_idx]

            for inst_id in untracked_ids:
                train_id = online_to_train_id.get(inst_id, -1)
                if train_id == -1:
                    print(f"[Dynamic INIT] Online Instance {inst_id} not found in training map. Skipping.")
                    continue
                Y_ind = active_idx[active_inst_ids == train_id]
                if len(Y_ind) > 0:
                    instance_particles[inst_id] = Y_ind
                    if inst_id not in committed_cumulative:
                        committed_cumulative[inst_id] = np.eye(4, dtype=np.float64)
                        committed_frame_transforms[inst_id] = [
                            np.eye(4, dtype=np.float64) for _ in range(len(frame_indices))
                        ]
                    instance_centroids[inst_id] = neural_points.map_points[Y_ind].detach().cpu().numpy().mean(axis=0)
                    print(f"[Dynamic INIT] Instance {inst_id} (Train ID: {train_id}): assigned {len(Y_ind)} particles.")
                else:
                    print(f"[Dynamic INIT] Instance {inst_id} (Train ID: {train_id}): No particles found in map.")

        # --- Extract keypoints and run CoTracker per view --- #
        chunk_results: dict = {}
        any_valid_keypoints = False
        anchor_t = buffer_size - 2
        target_t = buffer_size - 1

        for view in views:
            buf = buffers[view]
            frames_np = np.stack(buf["rgb"])
            chunk_depths = torch.from_numpy(np.stack(buf["depth"])).to(device)
            chunk_poses = torch.from_numpy(np.stack(buf["poses"])).to(device)
            chunk_poses_c2w = torch.linalg.inv(chunk_poses)[:, :3, :4]
            chunk_segs = torch.from_numpy(np.stack(buf["seg"]).astype(np.int64)).to(device)

            seg_first = buf["seg"][0]
            view_keypoints: list[tuple[float, float]] = []
            view_kp_inst_ids: list[int] = []

            for inst_id in instance_particles:
                mask = np.zeros_like(seg_first, dtype=np.uint8)
                mask[seg_first == inst_id] = 255
                if np.sum(mask) == 0:
                    continue
                kps = extract_keypoints_automatically(
                    frames_np[0], mask,
                    max_points=cfg.max_total_points_per_instance,
                    erosion_iters=cfg.erosion_iters,
                    quality_level=cfg.kp_quality_level,
                    min_distance=cfg.kp_min_distance,
                    block_size=cfg.kp_block_size,
                )
                remaining = cfg.max_total_points_per_instance - len(kps)
                if remaining > 0:
                    rand_kps = sample_random_mask_points(
                        mask, remaining,
                        erosion_iters=cfg.erosion_iters,
                        min_distance=cfg.kp_min_distance,
                        existing_points=kps,
                    )
                    kps = kps + rand_kps
                view_keypoints.extend(kps)
                view_kp_inst_ids.extend([inst_id] * len(kps))

            if not view_keypoints:
                chunk_results[view] = {
                    "valid": False,
                    "frames_np": frames_np,
                    "chunk_depths": chunk_depths,
                    "chunk_poses": chunk_poses,
                    "chunk_poses_c2w": chunk_poses_c2w,
                    "chunk_segs": chunk_segs,
                }
                continue

            any_valid_keypoints = True
            tracks_xy, visibilities, confidences = run_cotracker_offline(
                frames_np, view_keypoints, cotracker, device=device,
            )
            chunk_results[view] = {
                "valid": True,
                "frames_np": frames_np,
                "chunk_depths": chunk_depths,
                "chunk_poses": chunk_poses,
                "chunk_poses_c2w": chunk_poses_c2w,
                "chunk_segs": chunk_segs,
                "keypoints": view_keypoints,
                "kp_inst_ids": torch.tensor(view_kp_inst_ids, dtype=torch.long, device=device),
                "tracks_xy": tracks_xy,
                "visibilities": visibilities,
                "confidences": confidences,
                "colors_2d": np.random.randint(0, 255, (len(view_keypoints), 3), dtype=np.uint8),
            }

        # --- Fallback: retry views with under-tracked instances using 2x budget --- #
        # Only retry instances committed within the last fallback_lookback iterations.
        retry_instances: set[int] = set()
        recent_moved = {iid for iid, last_iter in moved_instances.items()
                        if iter_idx - last_iter <= cfg.fallback_lookback}
        if any_valid_keypoints and recent_moved:
            inst_ids_to_check = [iid for iid in instance_particles if iid in recent_moved]
            kp_counts = count_registration_correspondences_per_instance(
                chunk_results, views, inst_ids_to_check,
                anchor_t, target_t,
                cfg.visibility_threshold, cfg.confidence_threshold,
            )
            anchor_segmentations = {view: buffers[view]["seg"][anchor_t] for view in views}
            retry_instances, retry_views = select_fallback_retry_instances_and_views(
                views=views,
                inst_ids_to_check=inst_ids_to_check,
                kp_counts=kp_counts,
                min_observed_keypoints=cfg.min_observed_keypoints,
                anchor_segmentations=anchor_segmentations,
                required_visibility_view_substrings=("wrist",),
            )

            if retry_instances:
                doubled_budget = cfg.max_total_points_per_instance * 2
                halved_min_distance = max(1, cfg.kp_min_distance // 2)
                print(f"[FALLBACK] Retrying {len(retry_instances)} under-tracked instances "
                      f"with {doubled_budget} points (min_distance={halved_min_distance}): {sorted(retry_instances)}")

                for view in retry_views:
                    res = chunk_results[view]
                    buf = buffers[view]
                    frames_np = res.get("frames_np")
                    if frames_np is None:
                        frames_np = np.stack(buf["rgb"])
                    seg_first = buf["seg"][0]

                    view_keypoints: list[tuple[float, float]] = []
                    view_kp_inst_ids: list[int] = []

                    for inst_id in instance_particles:
                        mask = np.zeros_like(seg_first, dtype=np.uint8)
                        mask[seg_first == inst_id] = 255
                        if np.sum(mask) == 0:
                            continue
                        is_retry = inst_id in retry_instances
                        budget = doubled_budget if is_retry else cfg.max_total_points_per_instance
                        min_dist = halved_min_distance if is_retry else cfg.kp_min_distance
                        kps = extract_keypoints_automatically(
                            frames_np[0], mask,
                            max_points=budget,
                            erosion_iters=cfg.erosion_iters,
                            quality_level=cfg.kp_quality_level,
                            min_distance=min_dist,
                            block_size=cfg.kp_block_size,
                        )
                        remaining = budget - len(kps)
                        if remaining > 0:
                            rand_kps = sample_random_mask_points(
                                mask, remaining,
                                erosion_iters=cfg.erosion_iters,
                                min_distance=min_dist,
                                existing_points=kps,
                            )
                            kps = kps + rand_kps
                        view_keypoints.extend(kps)
                        view_kp_inst_ids.extend([inst_id] * len(kps))

                    if view_keypoints:
                        tracks_xy, visibilities, confidences = run_cotracker_offline(
                            frames_np, view_keypoints, cotracker, device=device,
                        )
                        chunk_results[view] = {
                            **{k: v for k, v in res.items()
                               if k not in ("valid", "keypoints", "kp_inst_ids",
                                            "tracks_xy", "visibilities", "confidences",
                                            "colors_2d", "_tracks_np", "_vis_np", "_conf_np")},
                            "valid": True,
                            "keypoints": view_keypoints,
                            "kp_inst_ids": torch.tensor(view_kp_inst_ids, dtype=torch.long, device=device),
                            "tracks_xy": tracks_xy,
                            "visibilities": visibilities,
                            "confidences": confidences,
                            "colors_2d": np.random.randint(0, 255, (len(view_keypoints), 3), dtype=np.uint8),
                        }

        # --- Register last frame in buffer --- #
        geometry_changed = False

        if frame_idx > start_frame:
            for inst_id in list(instance_particles.keys()):
                outcome = register_instance_for_frame(
                    inst_id, anchor_t, target_t, views, chunk_results, intrinsics,
                    instance_particles, neural_points,
                    committed_cumulative, instance_centroids,
                    cfg, device,
                    is_fallback_retry=(inst_id in retry_instances),
                )
                changed = outcome == "changed"
                if changed:
                    moved_instances[inst_id] = iter_idx
                geometry_changed = geometry_changed or changed

        # --- Render video frames --- #
        if cfg.save_video and vis is not None:
            img, pca_colors_arr = render_3d_frame(
                vis, pcd, neural_points, pca_colors_arr,
                excluded_train_ids_tensor, cfg.camera_extrinsic, geometry_changed,
            )
            writer.append_data(img)
            writer_2d.append_data(render_2d_frame(
                target_t, views, chunk_results,
                visibility_threshold=cfg.visibility_threshold,
                confidence_threshold=cfg.confidence_threshold,
            ))

        # --- Record frame transform state (fill skipped frames with latest pose) --- #
        last_recorded = frame_indices[-1] if frame_indices else start_frame - 1
        for fill_idx in range(last_recorded + 1, min(frame_idx + frame_step, end_frame)):
            frame_indices.append(fill_idx)
            for iid in committed_cumulative:
                committed_frame_transforms[iid].append(committed_cumulative[iid].copy())

    return pca_colors_arr


# --- Main --- #

def run_tracking(cfg: TrackingConfig) -> None:
    """Run the full tracking pipeline with the given configuration."""
    if cfg.save_video:
        import open3d as o3d

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # --- Load HDF5 Data --- #
    hdf5_file = h5py.File(cfg.data_path, "r")
    views = [v for v in cfg.views if v in hdf5_file]
    if not views:
        hdf5_file.close()
        raise ValueError("No valid views found in HDF5 file!")

    intrinsics = {}
    num_frames = float("inf")
    for view in views:
        grp = hdf5_file[view]
        n_frames = grp["rgb"].shape[0]
        num_frames = min(num_frames, n_frames)
        intr = grp["intrinsics"][:]
        intrinsics[view] = (float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2]))
        print(f"[INFO] View '{view}': found {n_frames} frames.")
    print(f"[INFO] Using views: {views}. Max common frames: {num_frames}")

    # --- Load Neural Point Map --- #
    neural_points = _load_neural_point_map(cfg, device)
    episode_name = os.path.splitext(os.path.basename(cfg.data_path))[0]
    if cfg.save_video:
        cfg.output_video_path = _resolve_tracking_video_path(cfg.output_video_path, episode_name)
        cfg.output_2d_video_path = _resolve_tracking_video_path(cfg.output_2d_video_path, episode_name)

    # --- Instance IDs --- #
    online_to_train_id, training_id_map, excluded_train_ids_tensor, target_instance_ids = \
        _build_instance_id_context(cfg, hdf5_file, neural_points, device)

    # --- PCA Colors (only needed for video) --- #
    if cfg.save_video:
        _, pca_colors_arr, _, _ = compute_pca_colors(neural_points, excluded_train_ids_tensor)
    else:
        pca_colors_arr = None

    # --- Capture initial state for HDF5 export --- #
    initial_points = neural_points.map_points.detach().cpu().numpy()
    initial_features = neural_points.map_features.detach().cpu().numpy()
    initial_instance_ids = neural_points.map_instance_ids.cpu().numpy()

    # --- Initialize Visualizer --- #
    if cfg.save_video:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Dynamic Map Update", width=cfg.vis_window_width, height=cfg.vis_window_height, visible=False)

        pcd = o3d.geometry.PointCloud()
        active_indices = get_active_foreground_indices(neural_points, excluded_train_ids_tensor)
        active_indices_np = active_indices.cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(neural_points.map_points[active_indices].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pca_colors_arr[active_indices_np])
        vis.add_geometry(pcd)

        apply_camera_extrinsic(vis, cfg.camera_extrinsic)

        writer = imageio.get_writer(cfg.output_video_path, fps=cfg.video_fps)
        writer_2d = imageio.get_writer(cfg.output_2d_video_path, fps=cfg.video_fps)

    # --- Load CoTracker --- #
    print("[INIT] Loading CoTracker model...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    cotracker.eval()

    start_frame = cfg.start_frame
    end_frame = num_frames if cfg.end_frame == -1 else min(cfg.end_frame, num_frames)
    print(f"[RUN] Starting Iterative Tracking from {start_frame} to {end_frame}...")

    instance_particles = {}
    committed_cumulative = {}
    frame_indices = []
    committed_frame_transforms = {}
    moved_instances: dict[int, int] = {}  # inst_id → last committed iteration index
    instance_centroids: dict[int, np.ndarray] = {}

    if not cfg.save_video:
        vis = pcd = writer = writer_2d = None

    pca_colors_arr = _run_streaming_loop(
        cfg, start_frame, end_frame, views, hdf5_file,
        instance_particles, intrinsics, cotracker, neural_points,
        online_to_train_id, target_instance_ids, excluded_train_ids_tensor,
        committed_cumulative, instance_centroids,
        frame_indices, committed_frame_transforms, moved_instances, device,
        pca_colors_arr=pca_colors_arr,
        vis=vis if cfg.save_video else None,
        pcd=pcd if cfg.save_video else None,
        writer=writer if cfg.save_video else None,
        writer_2d=writer_2d if cfg.save_video else None,
    )

    if cfg.save_video:
        writer.close()
        writer_2d.close()
        vis.destroy_window()
        print(f"[DONE] 3D Video saved to {cfg.output_video_path}")
        print(f"[DONE] 2D Video saved to {cfg.output_2d_video_path}")
    hdf5_file.close()

    # --- Save tracking results to HDF5 --- #
    if committed_frame_transforms:
        excluded_train_ids = set(excluded_train_ids_tensor.cpu().tolist())
        export_tracking_results(
            cfg, episode_name, initial_points, initial_features,
            initial_instance_ids, frame_indices, committed_frame_transforms,
            training_id_map, exclude_ids=None if cfg.save_all_instances else excluded_train_ids,
        )


def main():
    parser = argparse.ArgumentParser(description="Neural point tracking with iterative keypoint extraction.")
    parser.add_argument("--config", type=str, default="tracking/config/tracking.yaml", help="Path to tracking YAML config file.")
    parser.add_argument("--task", type=str, default=None, help="Task number (e.g. '0021'). Auto-loads task-specific config.")
    parser.add_argument("--data_path", type=str, default=None, help="Override data_path from config.")
    parser.add_argument("--run_dir", type=str, default=None, help="Override run_dir from config.")
    parser.add_argument("--start_frame", type=int, default=None, help="Override start_frame.")
    parser.add_argument("--end_frame", type=int, default=None, help="Override end_frame.")
    parser.add_argument("--save_all_instances", action="store_true", help="Save all instances (including excluded) to output HDF5.")
    args = parser.parse_args()

    # Load config: --task merges task-specific overlay on base config
    if args.task:
        cfg = load_task_config(args.task, base_config_path=args.config)
    elif args.config:
        cfg = TrackingConfig.from_yaml(args.config)
    else:
        cfg = TrackingConfig()

    # CLI overrides
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.run_dir is not None:
        cfg.run_dir = args.run_dir
    if args.start_frame is not None:
        cfg.start_frame = args.start_frame
    if args.end_frame is not None:
        cfg.end_frame = args.end_frame
    if args.save_all_instances:
        cfg.save_all_instances = True

    if not cfg.run_dir:
        parser.error("--run_dir is required (via config or CLI)")

    run_tracking(cfg)


if __name__ == "__main__":
    main()
