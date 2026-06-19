import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import omnigibson as og
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_INDICES_TO_NAMES,
    generate_basic_environment_config,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger
from tqdm import tqdm
from omnigibson.sensors import VisionSensor
from omnigibson.utils.python_utils import h5py_group_to_torch, recursively_convert_to_torch

from utils.geometry import calculate_intrinsics
from utils.pose import pose_to_extrinsic_cv, extrinsic_cv_to_pose

try:
    from gello.robots.sim_robot.og_teleop_utils import load_available_tasks, generate_robot_config
except ImportError:
    print("Warning: gello not found. JSON mode might fail if not configured correctly.")

# Global settings
log = create_module_logger(module_name="render_sampled_pose")
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_HQ_RENDERING = False
gm.HEADLESS = True
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128
gm.ENABLE_TRANSITION_RULES = False


def load_initial_episode_state(env, episode_id: int):
    """Load only the initial simulator state from a recorded HDF5 episode."""
    data_grp = env.input_hdf5["data"]
    traj_grp_name = f"demo_{episode_id}"
    assert traj_grp_name in data_grp, f"No valid episode with ID {episode_id} found!"

    traj = h5py_group_to_torch(data_grp[traj_grp_name])
    init_metadata = traj["init_metadata"]
    state = traj["state"]
    state_size = traj["state_size"]

    env.scene.restore(env.scene_file, update_initial_file=True)

    with og.sim.stopped():
        for attr, vals in init_metadata.items():
            assert len(vals) == env.scene.n_objects
        for i, obj in enumerate(env.scene.objects):
            for attr, vals in init_metadata.items():
                val = vals[i]
                setattr(obj, attr, val.item() if val.ndim == 0 else val)

    env.reset()

    if not env.include_robot_control:
        for robot in env.robots:
            robot.control_enabled = False

    og.sim.load_state(state[0, : int(state_size[0])], serialized=True)


class HDF5DataSaver:
    """Handles buffered HDF5 data saving for efficient I/O."""

    def __init__(self, output_path: str, flush_threshold: int = 100, fixed_instance_id_map: Optional[dict] = None):
        self.output_path = output_path
        self.flush_threshold = flush_threshold
        self.output_file: Optional[h5py.File] = None
        self.data_buffers = {}
        self.buffer_size = 0
        self.cumulative_instance_id_map = {}
        # Fixed instance ID mapping: object_name -> fixed_id
        self.fixed_instance_id_map = fixed_instance_id_map
    
    def open(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.output_file = h5py.File(self.output_path, "w")
        self.data_buffers = {}
        self.buffer_size = 0
        self.cumulative_instance_id_map = {}
    
    def close(self):
        if self.output_file:
            self.flush_buffer()
            # Save instance id mappings
            # Always save cumulative_instance_id_map as original mapping (for reference)
            if self.cumulative_instance_id_map:
                self.output_file.attrs["original_instance_id_to_name"] = json.dumps(self.cumulative_instance_id_map)

            # Save fixed mapping as the main instance_id_to_name (used for segmentation)
            if self.fixed_instance_id_map is not None:
                # Convert fixed_instance_id_map (name -> id) to (id -> name) format
                fixed_id_to_name = {str(v): k for k, v in self.fixed_instance_id_map.items()}
                self.output_file.attrs["instance_id_to_name"] = json.dumps(fixed_id_to_name)
            elif self.cumulative_instance_id_map:
                # Fallback: use cumulative map if no fixed map provided
                self.output_file.attrs["instance_id_to_name"] = json.dumps(self.cumulative_instance_id_map)

            self.output_file.close()
            self.output_file = None
    
    def save_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg_instance_id: np.ndarray,
        pose: np.ndarray,
        info: Optional[dict] = None,
    ):
        """
        Save a single frame to the buffer.

        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            seg_instance_id: Segmentation instance ID image (H, W)
            pose: 4x4 extrinsic matrix in OpenCV format
            info: Optional observation info containing instance_id mapping
        """
        # RGB: Keep as uint8 (already compact)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Depth: Convert from meters (float32) to millimeters (uint16)
        # This reduces size by half (4 bytes -> 2 bytes)
        if depth.dtype != np.uint16:
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = (depth * 1000).clip(0, 65535).astype(np.uint16)

        # Pose: float32 for 4x4 matrix
        if pose.dtype != np.float32:
            pose = pose.astype(np.float32)

        # Update instance ID map from info
        if info and "seg_instance_id" in info:
            current_map = {int(k): v for k, v in info["seg_instance_id"].items()}
            self.cumulative_instance_id_map.update(current_map)

        # Remap instance IDs to fixed IDs if mapping is provided
        if self.fixed_instance_id_map is not None and info and "seg_instance_id" in info:
            # Build mapping from original ID -> fixed ID for this frame
            original_to_fixed = {}
            for orig_id_str, obj_name in info["seg_instance_id"].items():
                orig_id = int(orig_id_str)
                if obj_name in self.fixed_instance_id_map:
                    original_to_fixed[orig_id] = self.fixed_instance_id_map[obj_name]
                else:
                    # Object not in fixed mapping, use a sentinel value (e.g., 65535 for unknown)
                    original_to_fixed[orig_id] = 65535

            # Apply remapping to segmentation image
            remapped_seg = np.zeros_like(seg_instance_id, dtype=np.uint16)
            for orig_id, fixed_id in original_to_fixed.items():
                remapped_seg[seg_instance_id == orig_id] = fixed_id
            seg_instance_id = remapped_seg

        # Segmentation ID: Use uint16 (matches replay_dataset.py format)
        if seg_instance_id.dtype != np.uint16:
            seg_instance_id = seg_instance_id.astype(np.uint16)
        
        # Add to buffers
        data_map = {
            "rgb": rgb,
            "depth": depth,
            "seg_instance_id": seg_instance_id,
            "poses": pose,
        }

        for key, data in data_map.items():
            if key not in self.data_buffers:
                self.data_buffers[key] = []
            self.data_buffers[key].append(data)
        
        self.buffer_size += 1
        
        # Check flush threshold
        if self.buffer_size >= self.flush_threshold:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Flush buffered data to HDF5 file."""
        if not self.output_file or not self.data_buffers:
            return
        
        for key, data_list in self.data_buffers.items():
            if not data_list:
                continue
            
            data_np = np.stack(data_list)
            
            if key in self.output_file:
                dset = self.output_file[key]
                dset.resize(dset.shape[0] + data_np.shape[0], axis=0)
                dset[-data_np.shape[0]:] = data_np
            else:
                maxshape = (None,) + data_np.shape[1:]
                # Choose compression based on data type
                compression = "lzf"  # Fast compression
                self.output_file.create_dataset(
                    key,
                    data=data_np,
                    maxshape=maxshape,
                    chunks=(1,) + data_np.shape[1:],
                    compression=compression,
                )
            
            data_list.clear()
        
        self.buffer_size = 0
    
    def save_intrinsics(self, intrinsics: np.ndarray):
        """Save camera intrinsics matrix."""
        if self.output_file:
            self.output_file.create_dataset("intrinsics", data=intrinsics.astype(np.float32))
            self.output_file.attrs["depth_scale"] = 1000.0  # mm


def render_episode(
    env,
    sensor,
    all_poses: np.ndarray,
    output_folder: str,
    task_id: int,
    demo_id: Optional[int],
    image_width: int,
    image_height: int,
    n_render_iterations: int,
    flush_threshold: int,
    num_samples: int,
    collision_radius: float = 0.02,
    seed: int = 0,
    mode: str = "hdf5",
    scene_id: Optional[int] = None,
    fixed_instance_id_map: Optional[dict] = None,
):
    """
    Render a single episode from poses using the provided environment.

    Args:
        env: The DataPlaybackWrapper environment (reused across episodes)
        sensor: The external camera sensor
        all_poses: Array of all available poses (N, 4, 4) extrinsic matrices
        output_folder: Path to the output folder
        task_id: Task ID
        demo_id: Demo/Episode ID (required for hdf5 mode)
        image_width: Output image width
        image_height: Output image height
        n_render_iterations: Number of render iterations per frame
        flush_threshold: Number of frames to buffer before flushing to HDF5
        num_samples: Target number of valid samples to save
        collision_radius: Radius for collision checking sphere (default: 0.1m)
        seed: Random seed for pose sampling
        mode: "hdf5" or "json"
        scene_id: Scene ID (required for json mode)
    """
    total_available_poses = len(all_poses)
    
    # Setup output path
    if mode == "hdf5":
        output_path = os.path.join(output_folder, f"task-{task_id:04d}", "train", f"episode_{demo_id:08d}.hdf5")
    else:
        # JSON mode
        # Path logic from scripts/render_pose_task.py: task-XXXX/eval/scene_{scene_id}.hdf5
        output_path = os.path.join(output_folder, f"task-{task_id:04d}", "eval", f"scene_{scene_id}.hdf5")

    
    # Skip if already exists
    if os.path.exists(output_path):
        log.info(f"Output already exists: {output_path}, skipping...")
        return
    
    if mode == "hdf5":
        # Find the episode with max samples and load only the initial scene state.
        episode_num_samples = [env.input_hdf5["data"][key].attrs["num_samples"] for key in env.input_hdf5["data"].keys()]
        episode_id = episode_num_samples.index(max(episode_num_samples))
        log.info(f"Loading scene state from episode {episode_id}")
        
        load_initial_episode_state(env, episode_id)
        robot = env.robots[0]
    else:
        # JSON mode: Environment is already initialized and state loaded in main
        # We just need to find the robot
        robot = env.scene.object_registry("name", "robot_r1")
        if not robot:
             # Fallback if name is different
             for obj in env.scene.objects:
                 if "robot" in obj.name.lower():
                     robot = obj
                     break

    # Move robot far away
    if robot:
        robot.set_position_orientation(position=np.array([-100.0, -100.0, 10.0]))
    
    # Calculate and save intrinsics
    horizontal_aperture = sensor.horizontal_aperture
    focal_length = sensor.focal_length
    intrinsics = calculate_intrinsics(image_width, image_height, horizontal_aperture, focal_length)
    
    # Initialize HDF5 saver
    saver = HDF5DataSaver(output_path, flush_threshold=flush_threshold, fixed_instance_id_map=fixed_instance_id_map)
    saver.open()
    saver.save_intrinsics(intrinsics)
    
    # Warm-up for collision checking and rendering
    for _ in range(3):
        og.sim.step()
    
    # Warm-up render pipeline (ensures depth buffer is initialized)
    log.info("Warming up render pipeline...")
    for _ in range(5):
        og.sim.render()
    # Do a dummy observation to fully initialize the sensor pipeline
    _ = sensor.get_obs()
    
    # Render loop with collision checking
    log.info(f"Starting rendering with collision checking (radius={collision_radius})...")
    log.info(f"Target: {num_samples} valid samples from {total_available_poses} available poses")
    
    timers = {"set_pose": 0, "render": 0, "get_obs": 0, "save": 0, "collision": 0}
    log_interval = 100
    
    # Set up random sampling from available poses
    np.random.seed(seed)
    pose_indices = np.arange(total_available_poses)
    np.random.shuffle(pose_indices)
    
    saved_count = 0
    checked_count = 0
    collision_count = 0
    pose_idx_cursor = 0
    
    pbar = tqdm(total=num_samples, desc="Rendering (valid)")
    
    while saved_count < num_samples:
        # Check if we've exhausted all poses
        if pose_idx_cursor >= total_available_poses:
            log.warning(f"Exhausted all {total_available_poses} poses. Only saved {saved_count}/{num_samples} valid samples.")
            break
        
        # Get next pose
        pose_matrix = all_poses[pose_indices[pose_idx_cursor]]
        pose_idx_cursor += 1
        checked_count += 1
        
        # Convert extrinsic matrix to position and orientation
        t0 = time.time()
        pos, orn = extrinsic_cv_to_pose(pose_matrix)
        timers["set_pose"] += time.time() - t0
        
        # Collision check using sphere overlap
        t0 = time.time()
        has_collision = og.sim.psqi.overlap_sphere_any(radius=collision_radius, pos=pos)
        timers["collision"] += time.time() - t0
        
        if has_collision:
            collision_count += 1
            continue
        
        # Set sensor pose
        t0 = time.time()
        sensor.set_position_orientation(position=pos, orientation=orn)
        timers["set_pose"] += time.time() - t0
        
        # Render (no physics simulation)
        t0 = time.time()
        for _ in range(n_render_iterations):
            og.sim.render()
        timers["render"] += time.time() - t0
        
        # Get observations and extract data
        t0 = time.time()
        obs, info = sensor.get_obs()
        rgb = obs["rgb"][..., :3].cpu().numpy()  # (H, W, 3)
        depth = obs["depth_linear"].cpu().numpy().squeeze()  # (H, W)
        seg_instance_id = obs["seg_instance_id"].cpu().numpy().squeeze()  # (H, W)

        # Get actual pose after setting (for accuracy)
        cur_pos, cur_orn = sensor.get_position_orientation()
        extrinsic_cv = pose_to_extrinsic_cv(cur_pos.cpu().numpy(), cur_orn.cpu().numpy())
        timers["get_obs"] += time.time() - t0

        # Save frame
        t0 = time.time()
        saver.save_frame(rgb, depth, seg_instance_id, extrinsic_cv, info)
        timers["save"] += time.time() - t0
        
        saved_count += 1
        pbar.update(1)
        
        # Log timing
        if saved_count % log_interval == 0:
            log.info(
                f"Saved {saved_count}/{num_samples} (checked={checked_count}, collisions={collision_count}): "
                f"SetPose={timers['set_pose']/log_interval:.4f}s, "
                f"Collision={timers['collision']/log_interval:.4f}s, "
                f"Render={timers['render']/log_interval:.4f}s, "
                f"GetObs={timers['get_obs']/log_interval:.4f}s, "
                f"Save={timers['save']/log_interval:.4f}s"
            )
            timers = {k: 0 for k in timers}
    
    pbar.close()
    log.info(f"Rendering finished: {saved_count} valid samples saved, {collision_count} poses filtered due to collision")
    
    # Close HDF5
    saver.close()
    
    if saved_count < num_samples:
        log.warning(f"Could not reach target {num_samples} samples. Only {saved_count} valid poses found.")
    
    log.info(f"Rendering complete. Data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render images from sampled poses.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/behavior-1k/2025-challenge-rawdata",
        help="Path to the folder containing task-XXXX folders with raw HDF5 demos"
    )
    parser.add_argument(
        "--poses_file",
        type=str,
        default=None,
        help="Path to the npy file containing all sampled poses. If not specified, uses data/sampled_pose/task-XXXX/all_poses.npy"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/mapping_dataset",
        help="Path to the output folder for rendered HDF5 files"
    )
    parser.add_argument("--task_id", type=int, required=True, help="Task ID")
    parser.add_argument("--demo_id", type=int, default=None, help="Demo ID (required for hdf5 mode)")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of poses to sample from the poses file")
    parser.add_argument("--image_height", type=int, default=480, help="Output image height")
    parser.add_argument("--image_width", type=int, default=480, help="Output image width")
    parser.add_argument("--n_render_iterations", type=int, default=5, help="Number of render iterations per frame")
    parser.add_argument("--flush_threshold", type=int, default=100, help="Buffer size before flushing to HDF5")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for pose sampling")
    parser.add_argument("--collision_radius", type=float, default=0.03, help="Radius for collision checking sphere")
    
    # New arguments
    parser.add_argument("--mode", type=str, default="hdf5", choices=["hdf5", "json"], help="Mode: hdf5 or json")
    parser.add_argument("--scene_id", type=int, default=None, help="Scene ID (required for json mode)")
    parser.add_argument("--json_path", type=str, default=None, help="Path to JSON state file (required for json mode)")
    parser.add_argument("--instance_id_map", type=str, default=None, help="Path to fixed instance ID mapping JSON file. If not specified, uses data/instance_id_map/task-XXXX/instance_id_map.json")

    args = parser.parse_args()

    # Set default poses_file based on task_id if not specified
    if args.poses_file is None:
        args.poses_file = f"data/sampled_pose/task-{args.task_id:04d}/all_poses.npy"
        log.info(f"Using default poses file: {args.poses_file}")

    # Set default instance_id_map based on task_id if not specified
    if args.instance_id_map is None:
        args.instance_id_map = f"data/instance_id_map/task-{args.task_id:04d}/instance_id_map.json"
        log.info(f"Using default instance ID map: {args.instance_id_map}")

    # Load fixed instance ID mapping if provided
    fixed_instance_id_map = None
    if args.instance_id_map and os.path.exists(args.instance_id_map):
        with open(args.instance_id_map, "r") as f:
            fixed_instance_id_map = json.load(f)
        log.info(f"Loaded fixed instance ID mapping with {len(fixed_instance_id_map)} entries from {args.instance_id_map}")
    else:
        log.warning(f"Instance ID mapping file not found: {args.instance_id_map}")

    task_name = TASK_INDICES_TO_NAMES[args.task_id]
    
    # Determine ID for seeding and checks
    if args.mode == "hdf5":
        if args.demo_id is None:
            log.error("demo_id is required for hdf5 mode")
            return
        unique_id = args.demo_id
    else:
        if args.scene_id is None:
            log.error("scene_id is required for json mode")
            return
        unique_id = args.scene_id

    # Check if output already exists
    if args.mode == "hdf5":
        final_output_path = os.path.join(args.output_folder, f"task-{args.task_id:04d}", "train", f"episode_{unique_id:08d}.hdf5")
    else:
        final_output_path = os.path.join(args.output_folder, f"task-{args.task_id:04d}", "eval", f"scene_{unique_id}.hdf5")

    if os.path.exists(final_output_path):
        log.info(f"Output already exists: {final_output_path}, skipping...")
        return
    
    # Load poses from single npy file
    if not os.path.exists(args.poses_file):
        log.error(f"Poses file not found: {args.poses_file}")
        return
    
    all_poses = np.load(args.poses_file)  # (N, 4, 4)
    log.info(f"Loaded {len(all_poses)} poses from {args.poses_file}")
    
    # Setup seed and number of samples
    seed = args.seed if args.seed is not None else unique_id
    num_samples = args.num_samples
    
    log.info(f"Target: {num_samples} valid samples from {len(all_poses)} available poses")
    log.info(f"Collision radius: {args.collision_radius}m")
    
    if args.mode == "hdf5":
        # HDF5 Mode Logic
        input_dir = os.path.join(args.data_folder, f"task-{args.task_id:04d}")
        input_hdf5_path = os.path.join(input_dir, f"episode_{unique_id:08d}.hdf5")
        if not os.path.exists(input_hdf5_path):
            log.error(f"Input HDF5 not found: {input_hdf5_path}")
            return
        
        # Find full scene file
        task_scene_file_folder = os.path.join(
            os.path.dirname(os.path.dirname(og.__path__[0])), "joylo", "sampled_task", task_name
        )
        full_scene_file = None
        if os.path.exists(task_scene_file_folder):
            for file in os.listdir(task_scene_file_folder):
                if file.endswith(".json") and "partial_rooms" not in file:
                    full_scene_file = os.path.join(task_scene_file_folder, file)
                    break
        
        if full_scene_file is None:
            log.error(f"No full scene file found in {task_scene_file_folder}")
            return
        
        log.info(f"Using scene file: {full_scene_file}")
        
        # Robot sensor config
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb"],
                "sensor_kwargs": {
                    "image_height": 128,
                    "image_width": 128,
                },
            },
        }
        
        # External sensor config
        modalities = ["rgb", "depth_linear", "seg_instance_id"]
        external_sensors_config = [{
            "name": "external_camera",
            "sensor_type": "VisionSensor",
            "modalities": modalities,
            "sensor_kwargs": {
                "image_height": args.image_height,
                "image_width": args.image_width,
            },
            "position": [0, 0, 1.5],
            "orientation": [0, 0, 0, 1],
        }]
        
        dummy_output = f"/tmp/dummy_wrapper_output_{unique_id}.hdf5"
        
        env_wrapper = None
        try:
            log.info(f"Initializing Environment (HDF5 mode)...")
            env_wrapper = DataPlaybackWrapper.create_from_hdf5(
                input_path=input_hdf5_path,
                output_path=dummy_output,
                compression={"compression": "lzf"},
                robot_obs_modalities=["proprio"],
                robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
                robot_sensor_config=robot_sensor_config,
                external_sensors_config=external_sensors_config,
                n_render_iterations=args.n_render_iterations,
                flush_every_n_traj=1,
                flush_every_n_steps=500,
                include_robot_control=False,
                include_contacts=False,
                full_scene_file=full_scene_file,
            )
            sensor = env_wrapper._external_sensors["external_camera"]
            log.info("Environment initialized successfully")
            
            render_episode(
                env=env_wrapper,
                sensor=sensor,
                all_poses=all_poses,
                output_folder=args.output_folder,
                task_id=args.task_id,
                demo_id=unique_id,
                image_width=args.image_width,
                image_height=args.image_height,
                n_render_iterations=args.n_render_iterations,
                flush_threshold=args.flush_threshold,
                num_samples=num_samples,
                collision_radius=args.collision_radius,
                seed=seed,
                mode="hdf5",
                fixed_instance_id_map=fixed_instance_id_map,
            )

        except Exception as e:
            log.error(f"Failed to process demo {unique_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if env_wrapper is not None:
                for hdf5_attr in ("input_hdf5", "hdf5_file"):
                    hdf5_file = getattr(env_wrapper, hdf5_attr, None)
                    if hdf5_file is not None:
                        try:
                            hdf5_file.close()
                        except Exception:
                            pass
            if os.path.exists(dummy_output):
                try:
                    os.remove(dummy_output)
                except Exception:
                    pass
            og.shutdown()

    else:
        # JSON Mode Logic
        if "load_available_tasks" not in globals():
            log.error("gello not found (load_available_tasks missing). Cannot run JSON mode.")
            return

        if not args.json_path or not os.path.exists(args.json_path):
            log.error(f"JSON path not found: {args.json_path}")
            return
        
        log.info(f"Initializing Environment (JSON mode) for task {task_name}, scene {args.scene_id}...")
        
        try:
            available_tasks = load_available_tasks()
            task_cfg = available_tasks[task_name][0]
            cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
            
            # Setup robot config (minimal)
            cfg["robots"] = [
                generate_robot_config(
                    task_name=task_name,
                    task_cfg=task_cfg,
                )
            ]
            cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
            cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
            
            # Create Environment
            env = og.Environment(configs=cfg)
            
            # Load state from JSON
            with open(args.json_path, "r") as f:
                tro_state = recursively_convert_to_torch(json.load(f))
            
            log.info("Loading state from JSON...")
            for tro_key, state_data in tro_state.items():
                if tro_key == "robot_poses":
                    # We skip robot pose as we move it away anyway, but we can load it for completeness
                    pass
                else:
                    if tro_key in env.task.object_scope:
                        env.task.object_scope[tro_key].load_state(state_data, serialized=False)
                    else:
                        log.warning(f"Object {tro_key} from JSON not found in task object scope.")
            
            # Stabilization loop (borrowed from eval.py)
            log.info("Stabilizing scene...")
            for _ in range(25):
                og.sim.step_physics()
                for entity in env.task.object_scope.values():
                    if hasattr(entity, "keep_still"):
                        entity.keep_still()
            
            # Additional steps to ensure rendering buffers are ready
            for _ in range(5):
                og.sim.render()
            
            # Create External Camera manually
            log.info("Creating external camera...")
            camera_prim_path = "/external_camera"
            
            modalities = ["rgb", "depth_linear", "seg_instance_id"]
            sensor = VisionSensor(
                relative_prim_path=camera_prim_path,
                name="external_camera",
                modalities=modalities,
                image_height=args.image_height,
                image_width=args.image_width,
            )
            sensor.load(env.scene)
            sensor.initialize()
            
            render_episode(
                env=env,
                sensor=sensor,
                all_poses=all_poses,
                output_folder=args.output_folder,
                task_id=args.task_id,
                demo_id=None,
                scene_id=unique_id,
                image_width=args.image_width,
                image_height=args.image_height,
                n_render_iterations=args.n_render_iterations,
                flush_threshold=args.flush_threshold,
                num_samples=num_samples,
                collision_radius=args.collision_radius,
                seed=seed,
                mode="json",
                fixed_instance_id_map=fixed_instance_id_map,
            )
            
        except Exception as e:
            log.error(f"Failed to process scene {unique_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            og.shutdown()

    log.info("Processing complete.")


if __name__ == "__main__":
    main()
