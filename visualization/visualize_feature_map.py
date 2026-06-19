"""Visualize env + robot neural points with fixed PCA coloring (Open3D).

Keyboard controls:
    T / R  : Frame step +/- 100 (objects + robot)
    F / D  : Frame step +/- 1 (objects + robot)
    P      : Print camera extrinsic
    S      : Save screenshot
    Q      : Quit

Usage:
    # Default camera viewpoint is always applied
    python visualization/visualize_feature_map.py \
        --tracking_hdf5 data/exported_neural_points/task-0021/train/episode_00212800.hdf5 \
        --include_robot_model

    # Override camera: paste the P key output directly
    python visualization/visualize_feature_map.py \
        --tracking_hdf5 data/exported_neural_points/task-0021/train/episode_00212800.hdf5 \
        --camera_extrinsic 9.98273504e-01,-4.21909768e-02,...

"""

import argparse
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import open3d as o3d
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.robot import (
    URDF_PATH,
    RobotSurfaceSampler,
    load_robot_states,
    state_to_urdf_cfg,
    state_to_base_transform_matrix,
)
from utils.visualization import TorchPCA

# task-0021 default camera extrinsic
DEFAULT_CAMERA_EXTRINSIC = (
    "9.98334997e-01,5.13217264e-02,-2.63308623e-02,-5.78154722e-01,"
    "-3.33804146e-03,-4.04314050e-01,-9.14614130e-01,6.06912437e-02,"
    "-5.75855137e-02,9.13179188e-01,-4.03469552e-01,4.09359130e+00,"
    "0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00"
)
DEFAULT_POINT_SIZE = 5.0
COLOR_MODE_NAME = "PCs (2,3,4)"
PCA_SKIP_COMPONENTS = 2
DEFAULT_PCA_PERMUTATION = (1, 0, 2)
DEFAULT_PCA_PERMUTATION_LABEL = "GRB"


def compute_pca_colors_higher(
    features: torch.Tensor,
    skip_components: int = 3,
    n_components: int = 3,
    whiten: bool = True,
    quantile_low: float = 0.02,
    quantile_high: float = 0.98,
) -> np.ndarray:
    """Compute RGB colors from PCA components after skipping leading PCs."""
    total_components = skip_components + n_components
    pca = TorchPCA(n_components=total_components, whiten=whiten)
    pca_features = pca.fit_transform(features.detach())

    colors = pca_features[:, skip_components:total_components]
    q_low = np.quantile(colors, quantile_low, axis=0)
    q_high = np.quantile(colors, quantile_high, axis=0)
    colors = np.clip(colors, q_low, q_high)
    return (colors - q_low) / (q_high - q_low + 1e-8)


def load_tracking_hdf5(path: str) -> dict:
    """Load tracking results from HDF5."""
    data = {}
    with h5py.File(path, "r") as hf:
        data["initial_points"] = hf["initial_points"][:].astype(np.float32)
        data["initial_features"] = hf["initial_features"][:].astype(np.float32)
        data["initial_instance_ids"] = hf["initial_instance_ids"][:].astype(np.int64)
        data["frame_indices"] = hf["frame_indices"][:].astype(np.int64)

        transforms = {}
        for key in hf["transforms"]:
            transforms[int(key)] = hf["transforms"][key][:].astype(np.float32)
        data["transforms"] = transforms

        id_to_name_str = hf.attrs.get("instance_id_to_name", "{}")
        data["instance_id_to_name"] = json.loads(id_to_name_str)

    n = data["initial_points"].shape[0]
    num_inst = len(data["transforms"])
    print(f"[LOAD] {n} points, {len(data['frame_indices'])} frames, {num_inst} tracked instances")
    return data


def reconstruct_points_at_frame(data: dict, frame_idx: int) -> np.ndarray:
    """Reconstruct all point positions at a given frame index.

    Applies cumulative SE(3) transforms per tracked instance to the initial
    point positions. Non-tracked points remain at their initial positions.

    Args:
        data: Output of ``load_tracking_hdf5``.
        frame_idx: Index into the frame dimension (0-based into frame_indices).

    Returns:
        (N, 3) float32 array of reconstructed point positions.
    """
    points = data["initial_points"].copy()
    instance_ids = data["initial_instance_ids"]

    for inst_id, tf_all in data["transforms"].items():
        T = tf_all[frame_idx]  # (4, 4)
        R = T[:3, :3]
        t_vec = T[:3, 3]

        mask = instance_ids == inst_id
        if not mask.any():
            continue
        points[mask] = points[mask] @ R.T + t_vec

    return points


def resolve_parquet_path(tracking_hdf5: str, parquet_arg: str | None) -> Path:
    if parquet_arg:
        p = Path(parquet_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Parquet file not found: {p}")

    hdf5_path = Path(tracking_hdf5)
    episode_name = hdf5_path.stem
    task_dir = hdf5_path.parent.parent.name

    candidate = (
        project_root / "data" / "behavior-1k" / "2025-challenge-demos"
        / "data" / task_dir / f"{episode_name}.parquet"
    )
    if candidate.exists():
        print(f"[ROBOT] Auto-resolved parquet: {candidate}")
        return candidate

    raise FileNotFoundError(
        f"Robot parquet not found: {candidate}\n"
        "Download the matching BEHAVIOR-1K parquet file or use --parquet to specify."
    )


def resolve_model_dir(tracking_hdf5: str, model_dir_arg: str | None) -> Path:
    if model_dir_arg:
        p = Path(model_dir_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model directory not found: {p}")

    hdf5_path = Path(tracking_hdf5)
    task_dir = hdf5_path.parent.parent.name

    candidate = project_root / "data" / "map_models" / task_dir
    robot_pt = candidate / "neural_points" / "robot" / "robot_neural_points.pt"
    if robot_pt.exists():
        print(f"[ROBOT] Auto-resolved model dir: {candidate}")
        return candidate

    raise FileNotFoundError(
        f"Robot neural points not found: {candidate}/neural_points/robot/\n"
        "Use --model_dir to specify."
    )


def parse_camera_extrinsic(extrinsic_arg: str | None) -> np.ndarray:
    extrinsic_str = extrinsic_arg if extrinsic_arg is not None else DEFAULT_CAMERA_EXTRINSIC
    vals = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", extrinsic_str)]
    if len(vals) != 16:
        raise ValueError(f"--camera_extrinsic needs 16 values, got {len(vals)}")
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def apply_camera_extrinsic(vis, camera_extrinsic: np.ndarray) -> None:
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = camera_extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize env + robot neural points with fixed PCs (2,3,4) coloring (Open3D)."
    )
    parser.add_argument("--tracking_hdf5", type=str, required=True)
    parser.add_argument("--include_robot_model", action="store_true")
    parser.add_argument("--parquet", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument(
        "--camera_extrinsic", type=str, default=None,
        help="Override camera extrinsic: 16 comma-separated floats (row-major 4x4). "
             "Default viewpoint is always applied. Press P to print current extrinsic.",
    )
    parser.add_argument("--save_dir", type=str, default="screenshots",
                        help="Directory for screenshots (default: screenshots/)")
    return parser


@dataclass
class ViewerContext:
    data: dict
    origin: np.ndarray
    env_colors: np.ndarray
    env_pcd: o3d.geometry.PointCloud
    n_tracking_frames: int
    n_total_steps: int
    frame_step: int = 0
    robot_states: Any = None
    robot_sampler: RobotSurfaceSampler | None = None
    robot_colors: np.ndarray | None = None
    robot_pcd: o3d.geometry.PointCloud | None = None

    @property
    def has_robot(self) -> bool:
        return (
            self.robot_states is not None
            and self.robot_sampler is not None
            and self.robot_colors is not None
            and self.robot_pcd is not None
        )


def print_instance_counts(data: dict) -> None:
    id_to_name = data["instance_id_to_name"]
    instance_ids = data["initial_instance_ids"]

    print(f"\n[INSTANCES] Per-instance point counts:")
    for uid in np.unique(instance_ids):
        name = id_to_name.get(str(uid), "unknown")
        count = int((instance_ids == uid).sum())
        print(f"  ID {uid:4d}: {count:6d} pts  ({name})")
    print(f"  Total: {len(instance_ids)} pts\n")


def make_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    update_point_cloud(pcd, points, colors)
    return pcd


def update_point_cloud(
    pcd: o3d.geometry.PointCloud,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def apply_default_pca_permutation(colors: np.ndarray) -> np.ndarray:
    return colors[:, list(DEFAULT_PCA_PERMUTATION)]


def compute_fixed_pca_colors(
    env_features: np.ndarray,
    robot_points: Any | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    env_colors = compute_pca_colors_higher(
        torch.from_numpy(env_features), skip_components=PCA_SKIP_COMPONENTS,
    ).astype(np.float64)
    env_colors = apply_default_pca_permutation(env_colors)

    robot_colors = None
    if robot_points is not None:
        robot_colors = compute_pca_colors_higher(
            robot_points.features.detach(), skip_components=PCA_SKIP_COMPONENTS,
        ).astype(np.float64)
        robot_colors = apply_default_pca_permutation(robot_colors)

    return env_colors, robot_colors


def load_robot_inputs(args: argparse.Namespace) -> tuple[Any | None, RobotSurfaceSampler | None, Any | None]:
    if not args.include_robot_model:
        return None, None, None

    from mapping.representations.robot_neural_points import RobotNeuralPointMap

    parquet_path = resolve_parquet_path(args.tracking_hdf5, args.parquet)
    robot_states, _ = load_robot_states(parquet_path)

    model_dir = resolve_model_dir(args.tracking_hdf5, args.model_dir)
    robot_dir = model_dir / "neural_points" / "robot"

    sampler_path = robot_dir / "sampler.npz"
    if sampler_path.exists():
        sampler = RobotSurfaceSampler.load(sampler_path, URDF_PATH)
    else:
        sampler = RobotSurfaceSampler(URDF_PATH, voxel_size=0.02)
        sampler.save(sampler_path)
    print(f"[ROBOT] Sampler: {sampler.n_total_points} surface points")

    points_sd = torch.load(
        robot_dir / "robot_neural_points.pt", map_location="cpu", weights_only=True,
    )
    n_robot_pts = points_sd["features"].shape[0]
    robot_feat_dim = points_sd["features"].shape[1]
    robot_points = RobotNeuralPointMap(n_points=n_robot_pts, feature_dim=robot_feat_dim)
    robot_points.load_state_dict(points_sd)
    robot_points.eval()
    print(f"[ROBOT] Neural points: {n_robot_pts}, feature_dim={robot_feat_dim}")

    return robot_states, sampler, robot_points


def sample_robot_points(
    sampler: RobotSurfaceSampler,
    robot_states: Any,
    step: int,
    origin: np.ndarray,
) -> np.ndarray:
    robot_step = min(step, len(robot_states) - 1)
    cfg = state_to_urdf_cfg(robot_states[robot_step])
    base_tf = state_to_base_transform_matrix(robot_states[robot_step])
    points, _ = sampler.get_points(cfg, base_tf)
    return (points - origin).astype(np.float64)


def build_viewer_context(args: argparse.Namespace, data: dict) -> ViewerContext:
    robot_states, robot_sampler, robot_points = load_robot_inputs(args)
    env_colors, robot_colors = compute_fixed_pca_colors(data["initial_features"], robot_points)

    instance_ids = data["initial_instance_ids"]
    print(
        f"[PCA] Using fixed color mode: {COLOR_MODE_NAME} "
        f"({DEFAULT_PCA_PERMUTATION_LABEL}, {len(instance_ids)} env points)"
    )

    origin = data["initial_points"].mean(axis=0) if len(instance_ids) else np.zeros(3)
    env_points = (data["initial_points"] - origin).astype(np.float64)
    env_pcd = make_point_cloud(env_points, env_colors)

    robot_pcd = None
    if robot_states is not None and robot_sampler is not None and robot_colors is not None:
        initial_robot_step = min(int(data["frame_indices"][0]), len(robot_states) - 1)
        robot_points_xyz = sample_robot_points(robot_sampler, robot_states, initial_robot_step, origin)
        robot_pcd = make_point_cloud(robot_points_xyz, robot_colors)
        print(f"[ROBOT] {len(robot_points_xyz)} surface pts")

    n_tracking_frames = len(data["frame_indices"])
    n_total_steps = len(robot_states) if robot_states is not None else n_tracking_frames
    return ViewerContext(
        data=data,
        origin=origin,
        env_colors=env_colors,
        env_pcd=env_pcd,
        n_tracking_frames=n_tracking_frames,
        n_total_steps=n_total_steps,
        robot_states=robot_states,
        robot_sampler=robot_sampler,
        robot_colors=robot_colors,
        robot_pcd=robot_pcd,
    )


def refresh_view(vis, ctx: ViewerContext) -> bool:
    tracking_step = min(ctx.frame_step, ctx.n_tracking_frames - 1)
    env_points = reconstruct_points_at_frame(ctx.data, tracking_step)
    env_points = (env_points - ctx.origin).astype(np.float64)

    update_point_cloud(ctx.env_pcd, env_points, ctx.env_colors)
    vis.update_geometry(ctx.env_pcd)

    robot_count = 0
    if ctx.has_robot:
        robot_points = sample_robot_points(
            ctx.robot_sampler, ctx.robot_states, ctx.frame_step, ctx.origin,
        )
        update_point_cloud(ctx.robot_pcd, robot_points, ctx.robot_colors)
        vis.update_geometry(ctx.robot_pcd)
        robot_count = len(robot_points)

    print(
        f"[VIEW] frame={ctx.frame_step}/{ctx.n_total_steps - 1}  "
        f"color={COLOR_MODE_NAME}/{DEFAULT_PCA_PERMUTATION_LABEL}  "
        f"env={len(env_points)} robot={robot_count}"
    )
    return False


def step_frame(vis, ctx: ViewerContext, delta: int) -> bool:
    ctx.frame_step = min(max(ctx.frame_step + delta, 0), ctx.n_total_steps - 1)
    return refresh_view(vis, ctx)


def print_camera(vis) -> bool:
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    flat = ",".join(f"{v:.8e}" for v in np.asarray(params.extrinsic).flatten())
    print(f"\n[CAM] --camera_extrinsic {flat}")
    return False


def save_screenshot(vis, args: argparse.Namespace, ctx: ViewerContext) -> bool:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_id = Path(args.tracking_hdf5).stem
    mode_slug = COLOR_MODE_NAME.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "-")
    filename = save_dir / f"{episode_id}_{mode_slug}_t{ctx.frame_step:05d}.png"
    vis.capture_screen_image(str(filename), do_render=True)
    print(f"[SAVE] Screenshot saved: {filename}")
    return False


def configure_visualizer(vis, camera_extrinsic: np.ndarray) -> None:
    render_opt = vis.get_render_option()
    render_opt.point_size = DEFAULT_POINT_SIZE
    render_opt.background_color = np.array([1.0, 1.0, 1.0])
    apply_camera_extrinsic(vis, camera_extrinsic)


def add_geometries(vis, ctx: ViewerContext) -> None:
    vis.add_geometry(ctx.env_pcd)
    if ctx.robot_pcd is not None:
        vis.add_geometry(ctx.robot_pcd)


def print_controls(args: argparse.Namespace, ctx: ViewerContext) -> None:
    print(f"\n{'=' * 60}")
    print("  Open3D Keyboard Controls:")
    print(f"{'=' * 60}")
    print(f"    Color: {COLOR_MODE_NAME} / {DEFAULT_PCA_PERMUTATION_LABEL}")
    print(f"    Point size: {DEFAULT_POINT_SIZE:.0f}")
    print(f"    T  ->  Frame step +100 (objects + robot)")
    print(f"    R  ->  Frame step -100 (objects + robot)")
    print(f"    F  ->  Frame step +1 (objects + robot)")
    print(f"    D  ->  Frame step -1 (objects + robot)")
    print(f"    P  ->  Print camera extrinsic")
    print(f"    S  ->  Save screenshot to {args.save_dir}/")
    print(f"    Q  ->  Quit")
    print(f"{'=' * 60}\n")

    print(f"[VIS] Env: {len(ctx.env_pcd.points)} pts")
    if ctx.robot_pcd is not None:
        print(f"[VIS] Robot: {len(ctx.robot_pcd.points)} pts")


def run_interactive_viewer(
    args: argparse.Namespace,
    ctx: ViewerContext,
    camera_extrinsic: np.ndarray,
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Neural Points PCA", width=1920, height=1080)

    add_geometries(vis, ctx)
    vis.register_key_callback(ord("T"), lambda vis: step_frame(vis, ctx, 100))
    vis.register_key_callback(ord("R"), lambda vis: step_frame(vis, ctx, -100))
    vis.register_key_callback(ord("F"), lambda vis: step_frame(vis, ctx, 1))
    vis.register_key_callback(ord("D"), lambda vis: step_frame(vis, ctx, -1))
    vis.register_key_callback(ord("P"), print_camera)
    vis.register_key_callback(ord("S"), lambda vis: save_screenshot(vis, args, ctx))

    configure_visualizer(vis, camera_extrinsic)
    print_controls(args, ctx)

    vis.run()
    vis.destroy_window()


def main() -> None:
    args = build_arg_parser().parse_args()
    data = load_tracking_hdf5(args.tracking_hdf5)
    print_instance_counts(data)

    ctx = build_viewer_context(args, data)
    camera_extrinsic = parse_camera_extrinsic(args.camera_extrinsic)
    print("[CAM] Using camera extrinsic")

    run_interactive_viewer(args, ctx, camera_extrinsic)


if __name__ == "__main__":
    main()
