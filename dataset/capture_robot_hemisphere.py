from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from dataset.render_from_sampled_pose import HDF5DataSaver
from utils.geometry import calculate_intrinsics
from utils.pose import pose_to_extrinsic_cv, extrinsic_cv_to_pose, look_at_extrinsic_cv
from utils.robot import N_URDF_JOINTS, joint_qpos_to_urdf_cfg

# OmniGibson macros — must be set before importing og
from omnigibson.macros import gm

gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_HQ_RENDERING = False
gm.HEADLESS = True
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128
gm.ENABLE_TRANSITION_RULES = False

import omnigibson as og
from omnigibson.sensors import VisionSensor


# --- Plausible Pose Library --- #
#
# OmniGibson 28-DOF joint ordering (interleaved arms):
#   [0:6]   virtual base (always 0)
#   [6:10]  trunk: torso_joint1-4
#   [10:24] arms interleaved: [10,12,14,16,18,20,22]=L, [11,13,15,17,19,21,23]=R
#           j1=shoulder rot, j2=shoulder flex, j3=upper arm rot,
#           j4=elbow, j5=forearm rot, j6=wrist pitch, j7=wrist roll
#   [24:26] left gripper finger 1-2
#   [26:28] right gripper finger 1-2
#
# L arm j2: [-0.17, 3.14]  (positive = arm lifts up)
# R arm j2: [-3.14, 0.17]  (negative = arm lifts up, mirrored)
# Elbow j4: [-2.09, 0.35]  (negative = bent)


def _make_pose(
    trunk: tuple = (0, 0, 0, 0),
    left_arm: tuple = (0, 0, 0, 0, 0, 0, 0),
    right_arm: tuple = (0, 0, 0, 0, 0, 0, 0),
    left_grip: tuple = (0.02, 0.02),
    right_grip: tuple = (0.02, 0.02),
) -> np.ndarray:
    """Build a 28-DOF OmniGibson joint config from named groups."""
    cfg = np.zeros(N_URDF_JOINTS, dtype=np.float32)
    cfg[6:10] = trunk
    cfg[10:24:2] = left_arm      # left arm (even indices)
    cfg[11:24:2] = right_arm     # right arm (odd indices)
    cfg[24:26] = left_grip
    cfg[26:28] = right_grip
    return cfg


# fmt: off
PLAUSIBLE_POSES: List[np.ndarray] = [
    # ===== Neutral (3) =====
    _make_pose(),                                                                          # 0: default
    _make_pose(left_arm=(0, 0.3, 0, -0.4, 0, 0, 0),
               right_arm=(0, -0.3, 0, -0.4, 0, 0, 0)),                                   # 1: relaxed arms at sides
    _make_pose(left_arm=(0, 0.5, 0, -0.8, 0, 0, 0),
               right_arm=(0, -0.5, 0, -0.8, 0, 0, 0)),                                   # 2: arms hanging, elbows slightly bent

    # ===== Trunk-only (10) =====
    _make_pose(trunk=(0.8, 0, 0, 0)),                                                     # 3: twist R
    _make_pose(trunk=(-0.8, 0, 0, 0)),                                                    # 4: twist L
    _make_pose(trunk=(0, 1.5, 0, 0)),                                                     # 5: tilt R
    _make_pose(trunk=(0, -1.5, 0, 0)),                                                    # 6: tilt L
    _make_pose(trunk=(0, 0, 1.0, 0)),                                                     # 7: lean forward
    _make_pose(trunk=(0, 0, -1.0, 0)),                                                    # 8: lean back
    _make_pose(trunk=(0, 0, 0, 2.0)),                                                     # 9: rotate CW
    _make_pose(trunk=(0, 0, 0, -2.0)),                                                    # 10: rotate CCW
    _make_pose(trunk=(0.5, 0, 0.7, 0)),                                                   # 11: twist R + lean fwd
    _make_pose(trunk=(0, 1.0, 0.6, 0)),                                                   # 12: tilt R + lean fwd

    # ===== Single arm, no trunk (6) =====
    _make_pose(left_arm=(0, 1.57, 0, 0, 0, 0, 0)),                                       # 13: L arm horizontal
    _make_pose(left_arm=(0, 2.5, 0, 0, 0, 0, 0)),                                        # 14: L arm up high
    _make_pose(left_arm=(0, 1.0, 0, -1.0, 0, 0, 0)),                                     # 15: L arm forward bent
    _make_pose(right_arm=(0, -1.57, 0, 0, 0, 0, 0)),                                     # 16: R arm horizontal
    _make_pose(right_arm=(0, -2.5, 0, 0, 0, 0, 0)),                                      # 17: R arm up high
    _make_pose(right_arm=(0, -1.0, 0, -1.0, 0, 0, 0)),                                   # 18: R arm forward bent

    # ===== Both arms, no trunk (4) =====
    _make_pose(left_arm=(0, 1.57, 0, 0, 0, 0, 0),
               right_arm=(0, -1.57, 0, 0, 0, 0, 0)),                                     # 19: T-pose
    _make_pose(left_arm=(0, 2.5, 0, 0, 0, 0, 0),
               right_arm=(0, -2.5, 0, 0, 0, 0, 0)),                                      # 20: surrender
    _make_pose(left_arm=(0, 1.0, 0, -1.0, 0, 0, 0),
               right_arm=(0, -1.0, 0, -1.0, 0, 0, 0)),                                   # 21: both forward bent
    _make_pose(left_arm=(0, 1.57, 0, -1.57, 0, 0, 0),
               right_arm=(0, -1.57, 0, -1.57, 0, 0, 0)),                                 # 22: both L-shape

    # ===== Trunk + single arm (10) =====
    _make_pose(trunk=(0, 0, 0.6, 0),
               left_arm=(0, 1.0, 0, -0.8, 0, 0, 0)),                                     # 23: lean fwd + L reach
    _make_pose(trunk=(0, 0, 0.6, 0),
               right_arm=(0, -1.0, 0, -0.8, 0, 0, 0)),                                   # 24: lean fwd + R reach
    _make_pose(trunk=(0.7, 0, 0, 0),
               left_arm=(0, 2.0, 0, -0.5, 0, 0, 0)),                                     # 25: twist R + L arm up
    _make_pose(trunk=(-0.7, 0, 0, 0),
               right_arm=(0, -2.0, 0, -0.5, 0, 0, 0)),                                   # 26: twist L + R arm up
    _make_pose(trunk=(0, 1.2, 0, 0),
               left_arm=(0, 1.57, 0, 0, 0, 0, 0)),                                       # 27: tilt R + L arm out
    _make_pose(trunk=(0, -1.2, 0, 0),
               right_arm=(0, -1.57, 0, 0, 0, 0, 0)),                                     # 28: tilt L + R arm out
    _make_pose(trunk=(0, 0, -0.6, 0),
               left_arm=(0, 1.2, 0, -0.8, 0, 0, 0)),                                     # 29: lean back + L fwd
    _make_pose(trunk=(0, 0, -0.6, 0),
               right_arm=(0, -1.2, 0, -0.8, 0, 0, 0)),                                   # 30: lean back + R fwd
    _make_pose(trunk=(1.0, 0, 0.4, 0),
               left_arm=(0.3, 1.5, 0, -0.5, 0, 0, 0)),                                   # 31: deep twist R + L reach
    _make_pose(trunk=(-1.0, 0, 0.4, 0),
               right_arm=(-0.3, -1.5, 0, -0.5, 0, 0, 0)),                                # 32: deep twist L + R reach

    # ===== Trunk + both arms (12) =====
    _make_pose(trunk=(0, 0, 0.8, 0),
               left_arm=(0, 1.0, 0, -1.0, 0, 0, 0),
               right_arm=(0, -1.0, 0, -1.0, 0, 0, 0)),                                   # 33: lean fwd + push
    _make_pose(trunk=(0, 0, -0.7, 0),
               left_arm=(0, 2.0, 0, -0.3, 0, 0, 0),
               right_arm=(0, -2.0, 0, -0.3, 0, 0, 0)),                                   # 34: lean back + arms up
    _make_pose(trunk=(0.8, 0, 0, 0),
               left_arm=(0, 2.0, 0, -0.5, 0, 0, 0),
               right_arm=(0, -0.5, 0, -1.0, 0, 0, 0)),                                   # 35: twist R + L up R low
    _make_pose(trunk=(-0.8, 0, 0, 0),
               left_arm=(0, 0.5, 0, -1.0, 0, 0, 0),
               right_arm=(0, -2.0, 0, -0.5, 0, 0, 0)),                                   # 36: twist L + L low R up
    _make_pose(trunk=(0, 0, 1.0, 0),
               left_arm=(0, 0.3, 0, -0.5, 0, 0, 0),
               right_arm=(0, -0.3, 0, -0.5, 0, 0, 0)),                                   # 37: bow + arms at sides
    _make_pose(trunk=(0, 0, 0.5, 0),
               left_arm=(0.3, 1.5, 0, -0.3, 0, 0, 0),
               right_arm=(-0.3, -1.5, 0, -0.3, 0, 0, 0)),                                # 38: lean fwd + arms spread
    _make_pose(trunk=(0, 1.3, 0, 0),
               left_arm=(0, 2.0, 0, -0.5, 0, 0, 0),
               right_arm=(0, -0.5, 0, -1.2, 0, 0, 0)),                                   # 39: tilt R + L up R down
    _make_pose(trunk=(0, 0, 0.8, 0),
               left_arm=(0, 0.8, 0, -1.5, 0, 0, 0),
               right_arm=(0, -0.8, 0, -1.5, 0, 0, 0)),                                   # 40: deep lean + elbows bent
    _make_pose(trunk=(0.6, 0, 0, 0),
               left_arm=(0, 1.57, 0, 0, 0, 0, 0),
               right_arm=(0, -1.57, 0, 0, 0, 0, 0)),                                     # 41: twist R + T-pose
    _make_pose(trunk=(0, 0, 0, 1.5),
               left_arm=(0, 1.0, 0, -0.8, 0, 0, 0),
               right_arm=(0, -1.0, 0, -0.8, 0, 0, 0)),                                   # 42: rotate + arms fwd
    _make_pose(trunk=(0.5, 0, 0.5, 0),
               left_arm=(0, 2.0, 0, -0.8, 0, 0, 0),
               right_arm=(0, -0.5, 0, -1.2, 0, 0, 0)),                                   # 43: twist+lean + asym arms
    _make_pose(trunk=(-0.5, 0, 0.5, 0),
               left_arm=(0, 0.5, 0, -1.2, 0, 0, 0),
               right_arm=(0, -2.0, 0, -0.8, 0, 0, 0)),                                   # 44: twist+lean + asym arms (mirror)

    # ===== Gripper / wrist variations (5) =====
    _make_pose(trunk=(0, 0, 0.3, 0),
               left_arm=(0, 1.0, 0, -1.0, 0, 0, 0),
               right_arm=(0, -1.0, 0, -1.0, 0, 0, 0),
               left_grip=(0.05, 0.05), right_grip=(0.05, 0.05)),                          # 45: open grippers
    _make_pose(trunk=(0, 0, 0.3, 0),
               left_arm=(0, 1.0, 0, -1.0, 0, 0, 0),
               right_arm=(0, -1.0, 0, -1.0, 0, 0, 0),
               left_grip=(0.0, 0.0), right_grip=(0.0, 0.0)),                              # 46: closed grippers
    _make_pose(trunk=(0.3, 0, 0.2, 0),
               left_arm=(0, 1.0, 0, -1.0, 0, 0.8, 0.5),
               right_arm=(0, -1.0, 0, -1.0, 0, -0.8, -0.5)),                             # 47: wrists rotated
    _make_pose(trunk=(-0.3, 0, 0.2, 0),
               left_arm=(0, 1.0, 0, -1.0, 0.5, 0, 0),
               right_arm=(0, -1.0, 0, -1.0, -0.5, 0, 0)),                                # 48: forearms rotated
    _make_pose(trunk=(0, 0, 0.4, 0),
               left_arm=(0, 1.5, 0, -0.8, 0, 0.5, 0.8),
               right_arm=(0, -1.5, 0, -0.8, 0, -0.5, -0.8),
               left_grip=(0.05, 0.05), right_grip=(0.0, 0.0)),                            # 49: mixed grip + wrists
]
# fmt: on


def get_plausible_poses(n_states: Optional[int] = None, seed: int = 42) -> List[np.ndarray]:
    """Return curated plausible robot joint configurations.

    Args:
        n_states: If provided, randomly sample n_states poses. None = all.
        seed: Random seed for reproducible sampling.

    Returns:
        List of (28,) float32 arrays in OmniGibson joint ordering.
    """
    if n_states is None or n_states >= len(PLAUSIBLE_POSES):
        return list(PLAUSIBLE_POSES)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(PLAUSIBLE_POSES), size=n_states, replace=False)
    return [PLAUSIBLE_POSES[i] for i in sorted(indices)]


# --- Hemisphere Pose Generation --- #


def generate_hemisphere_poses(
    target: np.ndarray,
    radius: float,
    n_elevations: int,
    n_azimuths: int,
    elev_min_deg: float = 10.0,
    elev_max_deg: float = 70.0,
) -> np.ndarray:
    """Generate camera poses on a hemisphere looking at a target point.

    Args:
        target: (3,) center point cameras look at.
        radius: hemisphere radius in meters.
        n_elevations: number of elevation bands.
        n_azimuths: number of azimuth samples per band.
        elev_min_deg: minimum elevation angle in degrees.
        elev_max_deg: maximum elevation angle in degrees.

    Returns:
        (N, 4, 4) float32 array of extrinsic CV matrices, N = n_elev * n_azim.
    """
    elev_angles = np.linspace(np.radians(elev_min_deg), np.radians(elev_max_deg), n_elevations)
    azim_angles = np.linspace(0, 2 * np.pi, n_azimuths, endpoint=False)

    poses: List[np.ndarray] = []
    for elev in elev_angles:
        for azim in azim_angles:
            x = radius * np.cos(elev) * np.cos(azim) + target[0]
            y = radius * np.cos(elev) * np.sin(azim) + target[1]
            z = radius * np.sin(elev) + target[2]
            cam_pos = np.array([x, y, z])
            poses.append(look_at_extrinsic_cv(cam_pos, target))

    return np.stack(poses)


# --- Environment Setup --- #


def create_environment() -> og.Environment:
    """Create a minimal OmniGibson environment with R1Pro in an empty scene."""
    import torch as th

    controller_config = {
        "arm_left": {
            "name": "JointController",
            "motor_type": "position",
            "pos_kp": 150,
            "command_input_limits": None,
            "command_output_limits": None,
            "use_impedances": False,
            "use_delta_commands": False,
        },
        "arm_right": {
            "name": "JointController",
            "motor_type": "position",
            "pos_kp": 150,
            "command_input_limits": None,
            "command_output_limits": None,
            "use_impedances": False,
            "use_delta_commands": False,
        },
        "gripper_left": {
            "name": "MultiFingerGripperController",
            "mode": "smooth",
            "command_input_limits": "default",
            "command_output_limits": "default",
        },
        "gripper_right": {
            "name": "MultiFingerGripperController",
            "mode": "smooth",
            "command_input_limits": "default",
            "command_output_limits": "default",
        },
        "base": {
            "name": "HolonomicBaseJointController",
            "motor_type": "velocity",
            "vel_kp": 150,
            "command_input_limits": [-th.ones(3), th.ones(3)],
            "command_output_limits": [-th.tensor([0.75, 0.75, 1.0]), th.tensor([0.75, 0.75, 1.0])],
            "use_impedances": False,
        },
        "trunk": {
            "name": "JointController",
            "motor_type": "position",
            "pos_kp": 150,
            "command_input_limits": None,
            "command_output_limits": None,
            "use_impedances": False,
            "use_delta_commands": False,
        },
        "camera": {
            "name": "NullJointController",
        },
    }

    # Reset joint positions (R1Pro default: all zeros, fingers slightly open)
    reset_joint_pos = th.zeros(N_URDF_JOINTS, dtype=th.float32)
    reset_joint_pos[-4:] = 0.05  # open fingers

    cfg = {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {"type": "Scene"},
        "robots": [
            {
                "type": "R1Pro",
                "name": "robot_r1",
                "action_normalize": False,
                "controller_config": controller_config,
                "self_collisions": False,
                "obs_modalities": [],
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "grasping_mode": "assisted",
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": 128,
                            "image_width": 128,
                        },
                    },
                },
                "reset_joint_pos": reset_joint_pos,
            }
        ],
    }

    return og.Environment(configs=cfg)


def create_external_sensor(
    env: og.Environment,
    image_height: int,
    image_width: int,
) -> VisionSensor:
    """Create and initialize an external camera sensor."""
    sensor = VisionSensor(
        relative_prim_path="/external_camera",
        name="external_camera",
        modalities=["rgb", "depth_linear", "seg_instance_id"],
        image_height=image_height,
        image_width=image_width,
    )
    sensor.load(env.scene)
    sensor.initialize()
    return sensor


# --- Capture Loop --- #


def capture_multi_state(
    env: og.Environment,
    sensor: VisionSensor,
    robot,
    poses: np.ndarray,
    configs: List[np.ndarray],
    output_path: str,
    image_width: int,
    image_height: int,
    n_render_iterations: int,
    flush_threshold: int,
) -> None:
    """Capture hemisphere images for multiple robot states into a single HDF5.

    Args:
        env: OmniGibson environment.
        sensor: External camera VisionSensor.
        robot: OmniGibson robot object.
        poses: (M, 4, 4) extrinsic CV matrices (per-state camera poses).
        configs: List of N (28,) joint configurations in OmniGibson order.
        output_path: Path for the output HDF5 file.
        image_width: Output image width.
        image_height: Output image height.
        n_render_iterations: Number of render iterations per frame.
        flush_threshold: Frames to buffer before flushing to HDF5.
    """
    import torch as th

    n_states = len(configs)
    frames_per_state = len(poses)
    total_frames = n_states * frames_per_state
    print(f"[Capture] {n_states} states x {frames_per_state} poses = {total_frames} frames")
    print(f"[Capture] Saving to {output_path}")

    # Compute intrinsics
    intrinsics = calculate_intrinsics(
        image_width, image_height,
        sensor.horizontal_aperture, sensor.focal_length,
    )

    # Initialize saver
    saver = HDF5DataSaver(output_path, flush_threshold=flush_threshold)
    saver.open()
    saver.save_intrinsics(intrinsics)

    # Warm up renderer
    print("[Capture] Warming up render pipeline...")
    for _ in range(3):
        og.sim.step()
    for _ in range(5):
        og.sim.render()
    _ = sensor.get_obs()

    # Collect per-state robot data
    all_joint_positions = np.zeros((n_states, N_URDF_JOINTS), dtype=np.float32)
    all_urdf_cfgs = np.zeros((n_states, N_URDF_JOINTS), dtype=np.float32)
    all_base_transforms = np.zeros((n_states, 4, 4), dtype=np.float32)
    state_indices = np.zeros(total_frames, dtype=np.int32)

    for si in range(n_states):
        print(f"\n[Capture] === State {si}/{n_states} ===")

        # Set joint positions
        robot.set_joint_positions(th.tensor(configs[si], dtype=th.float32))
        for _ in range(10):
            og.sim.step_physics()
        for _ in range(5):
            og.sim.render()

        # Record actual robot state
        joint_pos = robot.get_joint_positions().cpu().numpy().astype(np.float32)
        all_joint_positions[si] = joint_pos
        all_urdf_cfgs[si] = joint_qpos_to_urdf_cfg(joint_pos)

        robot_pos, robot_orn = robot.get_position_orientation()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rotation.from_quat(robot_orn.cpu().numpy()).as_matrix()
        T[:3, 3] = robot_pos.cpu().numpy()
        all_base_transforms[si] = T

        # Fill state index
        frame_start = si * frames_per_state
        frame_end = frame_start + frames_per_state
        state_indices[frame_start:frame_end] = si

        # Render all poses for this state
        for pi in tqdm(range(frames_per_state), desc=f"State {si}"):
            pos, orn = extrinsic_cv_to_pose(poses[pi])
            sensor.set_position_orientation(position=pos, orientation=orn)

            for _ in range(n_render_iterations):
                og.sim.render()

            obs, info = sensor.get_obs()
            rgb = obs["rgb"][..., :3].cpu().numpy()
            depth = obs["depth_linear"].cpu().numpy().squeeze()
            seg_instance_id = obs["seg_instance_id"].cpu().numpy().squeeze()

            cur_pos, cur_orn = sensor.get_position_orientation()
            extrinsic_cv = pose_to_extrinsic_cv(cur_pos.cpu().numpy(), cur_orn.cpu().numpy())

            saver.save_frame(rgb, depth, seg_instance_id, extrinsic_cv, info)

    # Flush and close image data
    saver.close()

    # Append robot state metadata
    with h5py.File(output_path, "a") as f:
        f.create_dataset("state_index", data=state_indices, compression="lzf")
        f.create_dataset("robot_joint_positions", data=all_joint_positions,
                         chunks=(1,) + all_joint_positions.shape[1:], compression="lzf")
        f.create_dataset("robot_urdf_cfg", data=all_urdf_cfgs,
                         chunks=(1,) + all_urdf_cfgs.shape[1:], compression="lzf")
        f.create_dataset("robot_base_transform", data=all_base_transforms,
                         chunks=(1,) + all_base_transforms.shape[1:], compression="lzf")
        f.attrs["n_states"] = n_states
        f.attrs["frames_per_state"] = frames_per_state

    print(f"\n[Capture] Done. {total_frames} frames + {n_states} robot states -> {output_path}")


# --- Main --- #


def main():
    parser = argparse.ArgumentParser(description="Capture hemisphere images of R1Pro robot.")
    parser.add_argument("--output_path", type=str,
                        default="data/mapping_dataset/robot/robot_data.hdf5",
                        help="Output HDF5 file path")
    parser.add_argument("--n_states", type=int, default=len(PLAUSIBLE_POSES),
                        help="Number of robot poses to capture (from curated list)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for pose sampling")
    parser.add_argument("--radius", type=float, nargs="+", default=[0.8, 1.1, 1.4],
                        help="Hemisphere radii in meters (multiple values supported)")
    parser.add_argument("--n_elevations", type=int, default=10,
                        help="Number of elevation bands")
    parser.add_argument("--n_azimuths", type=int, default=12,
                        help="Number of azimuth samples per band")
    parser.add_argument("--elev_min", type=float, default=-30.0,
                        help="Minimum elevation angle in degrees")
    parser.add_argument("--elev_max", type=float, default=60.0,
                        help="Maximum elevation angle in degrees")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.7],
                        help="Look-at target point (x y z)")
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=480)
    parser.add_argument("--n_render_iterations", type=int, default=5,
                        help="Number of render iterations per frame")
    parser.add_argument("--flush_threshold", type=int, default=50,
                        help="Frames to buffer before flushing to HDF5")

    args = parser.parse_args()

    # Generate hemisphere poses for each radius
    target = np.array(args.target, dtype=np.float64)
    all_poses = []
    for r in args.radius:
        p = generate_hemisphere_poses(
            target=target,
            radius=r,
            n_elevations=args.n_elevations,
            n_azimuths=args.n_azimuths,
            elev_min_deg=args.elev_min,
            elev_max_deg=args.elev_max,
        )
        all_poses.append(p)
        print(f"[Main] Radius {r:.1f}m: {len(p)} poses")
    poses = np.concatenate(all_poses, axis=0)
    print(f"[Main] {len(poses)} poses/state x {args.n_states} states = {len(poses) * args.n_states} total frames")

    # Create environment
    print("[Main] Creating OmniGibson environment...")
    env = create_environment()
    robot = env.robots[0]

    # Get plausible joint configurations (in OmniGibson order)
    configs = get_plausible_poses(args.n_states, seed=args.seed)
    print(f"[Main] Using {len(configs)} curated plausible poses (seed={args.seed})")

    # Create external sensor
    sensor = create_external_sensor(env, args.image_height, args.image_width)

    # Capture all states
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    capture_multi_state(
        env=env,
        sensor=sensor,
        robot=robot,
        poses=poses,
        configs=configs,
        output_path=args.output_path,
        image_width=args.image_width,
        image_height=args.image_height,
        n_render_iterations=args.n_render_iterations,
        flush_threshold=args.flush_threshold,
    )

    og.shutdown()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
