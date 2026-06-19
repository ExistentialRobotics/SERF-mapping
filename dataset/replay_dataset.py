import argparse
import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES, TASK_INDICES_TO_NAMES
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

from utils.geometry import calculate_intrinsics
from utils.pose import pose_to_extrinsic_cv

# Head camera: 480x480, horizontal_aperture=40.0
# Wrist cameras: 480x480, horizontal_aperture=20.995
CAMERA_CONFIGS = {
    "head":        {"image_height": 480, "image_width": 480, "horizontal_aperture": 40.0, "focal_length": 17.0},
    "left_wrist":  {"image_height": 480, "image_width": 480, "horizontal_aperture": 20.995, "focal_length": 17.0},
    "right_wrist": {"image_height": 480, "image_width": 480, "horizontal_aperture": 20.995, "focal_length": 17.0},
}
MAX_IMAGE_SIZE = max(cfg["image_height"] for cfg in CAMERA_CONFIGS.values())
MIN_IMAGE_SIZE = min(cfg["image_height"] for cfg in CAMERA_CONFIGS.values())

# Global settings
log = create_module_logger(module_name="replay_obs")
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_HQ_RENDERING = False
gm.HEADLESS = True
gm.DEFAULT_VIEWER_WIDTH = 1280
gm.DEFAULT_VIEWER_HEIGHT = 720
gm.ENABLE_TRANSITION_RULES = False


class BehaviorDataPlaybackWrapper(DataPlaybackWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_rate = 1
        self.frame_idx = -1
        self.cumulative_instance_id_map = {}
        self.fixed_instance_id_map = None

        self.output_file = None
        self.data_buffers = {}
        self.buffer_size = 0
        self.flush_threshold = 500

    def init_hdf5(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.output_file = h5py.File(path, "w")
            
        self.data_buffers = {}
        self.buffer_size = 0

    def reset(self, *args, **kwargs):
        self.frame_idx = -1
        self.cumulative_instance_id_map = {}
        return super().reset(*args, **kwargs)

    def close_hdf5(self):
        if self.output_file:
            self.flush_buffer()
            if self.cumulative_instance_id_map:
                self.output_file.attrs["original_instance_id_to_name"] = json.dumps(self.cumulative_instance_id_map)

            if self.fixed_instance_id_map is not None:
                fixed_id_to_name = {str(v): k for k, v in self.fixed_instance_id_map.items()}
                self.output_file.attrs["instance_id_to_name"] = json.dumps(fixed_id_to_name)
            elif self.cumulative_instance_id_map:
                self.output_file.attrs["instance_id_to_name"] = json.dumps(self.cumulative_instance_id_map)

            self.output_file.close()
            self.output_file = None

    def save_frame_to_buffer(self, camera_name, rgb, depth, seg, pos, orn):
        # World to Camera Pose (OpenCV conventions)
        extrinsic_cv = pose_to_extrinsic_cv(pos.cpu().numpy(), orn.cpu().numpy())

        # Optimize Depth: Float32 (Meter) -> Uint16 (Millimeter)
        if depth.dtype != np.uint16:
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = (depth * 1000).astype(np.uint16)

        # Remap instance IDs to fixed IDs if mapping is provided
        if self.fixed_instance_id_map is not None and self.cumulative_instance_id_map:
            original_to_fixed = {}
            for orig_id, obj_name in self.cumulative_instance_id_map.items():
                orig_id_int = int(orig_id)
                if obj_name in self.fixed_instance_id_map:
                    original_to_fixed[orig_id_int] = self.fixed_instance_id_map[obj_name]
                else:
                    original_to_fixed[orig_id_int] = 65535
            remapped_seg = np.zeros_like(seg, dtype=np.uint16)
            for orig_id, fixed_id in original_to_fixed.items():
                remapped_seg[seg == orig_id] = fixed_id
            seg = remapped_seg

        data_map = {
            "rgb": rgb.astype(np.uint8),
            "depth": depth,
            "seg_instance_id": seg.astype(np.uint16),
            "poses": extrinsic_cv.astype(np.float32)
        }

        for key, val in data_map.items():
            full_key = f"{camera_name}/{key}"
            if full_key not in self.data_buffers:
                self.data_buffers[full_key] = []
            self.data_buffers[full_key].append(val)
        
        self.buffer_size += 1

    def flush_buffer(self):
        if not self.output_file or not self.data_buffers:
            return
        
        for full_key, data_list in self.data_buffers.items():
            if not data_list:
                continue
            data_np = np.stack(data_list)
            
            if full_key in self.output_file:
                dset = self.output_file[full_key]
                dset.resize(dset.shape[0] + data_np.shape[0], axis=0)
                dset[-data_np.shape[0]:] = data_np
            else:
                maxshape = (None,) + data_np.shape[1:]
                self.output_file.create_dataset(
                    full_key, 
                    data=data_np, 
                    maxshape=maxshape, 
                    chunks=(1,) + data_np.shape[1:], 
                    compression="lzf"
                )
            
            data_list.clear()
        self.buffer_size = 0

    def _process_obs(self, obs, info):
        self.frame_idx += 1
        robot = self.env.robots[0]
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []

        if 'obs_info' in info:
            for robot_obs_info in info['obs_info'].values():
                if not isinstance(robot_obs_info, dict):
                    continue
                for camera_obs in robot_obs_info.values():
                    if isinstance(camera_obs, dict) and 'seg_instance_id' in camera_obs:
                        self.cumulative_instance_id_map.update(camera_obs['seg_instance_id'])

        camera_map_inverse = {v: k for k, v in ROBOT_CAMERA_NAMES["R1Pro"].items()}

        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            sensor_name = camera_name.split("::")[1]
            if f"{camera_name}::seg_semantic" in obs:
                obs.pop(f"{camera_name}::seg_semantic")
            
            cam_pose = robot.sensors[sensor_name].get_position_orientation()
            
            if self.output_file and self.frame_idx % self.sampling_rate == 0:
                if f"{camera_name}::rgb" in obs:
                    rgb = obs[f"{camera_name}::rgb"][..., :3].cpu().numpy()
                    depth = obs[f"{camera_name}::depth_linear"].cpu().numpy().squeeze()
                    seg = obs[f"{camera_name}::seg_instance_id"].cpu().numpy().squeeze()
                    short_key = camera_map_inverse.get(camera_name, camera_name)
                    self.save_frame_to_buffer(short_key, rgb, depth, seg, cam_pose[0], cam_pose[1])

            cam_rel_poses.append(torch.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
            
            if self.buffer_size >= self.flush_threshold * len(ROBOT_CAMERA_NAMES["R1Pro"]):
                self.flush_buffer()

        obs["robot_r1::cam_rel_poses"] = torch.cat(cam_rel_poses, dim=-1)
        return obs
        
    def postprocess_traj_group(self, traj_grp):
        pass


def process_episode(
    env: BehaviorDataPlaybackWrapper,
    task_id: int,
    demo_id: int,
    camera_names: Dict[str, str],
    sampling_rate: int,
    output_folder: str,
):
    final_output_path = os.path.join(output_folder, f"task-{task_id:04d}", f"episode_{demo_id:08d}.hdf5")

    if os.path.exists(final_output_path):
        os.remove(final_output_path)

    env.init_hdf5(final_output_path)
    env.sampling_rate = sampling_rate

    num_samples = [env.input_hdf5["data"][key].attrs["num_samples"] for key in env.input_hdf5["data"].keys()]
    episode_id = num_samples.index(max(num_samples))
    log.info(f" >>> Replaying episode {episode_id}")

    camera_map_inverse = {v: k for k, v in camera_names.items()}
    for short_key, cam_name in camera_names.items():
        cam_cfg = CAMERA_CONFIGS[short_key]
        camera_sensor = env.robots[0].sensors[cam_name.split("::")[1]]
        camera_sensor.horizontal_aperture = cam_cfg["horizontal_aperture"]
        camera_sensor.focal_length = cam_cfg["focal_length"]
        # Only change resolution if different from creation default (MIN_IMAGE_SIZE).
        # Upsizing is safe; downsizing crashes IsaacSim's render graph annotator.
        if cam_cfg["image_height"] != MIN_IMAGE_SIZE or cam_cfg["image_width"] != MIN_IMAGE_SIZE:
            camera_sensor.image_height = cam_cfg["image_height"]
            camera_sensor.image_width = cam_cfg["image_width"]

        intrinsics = calculate_intrinsics(
            cam_cfg["image_width"], cam_cfg["image_height"],
            cam_cfg["horizontal_aperture"], cam_cfg["focal_length"],
        )

        if env.output_file:
            if short_key not in env.output_file:
                env.output_file.create_group(short_key)
            env.output_file[short_key].create_dataset("intrinsics", data=intrinsics)
            env.output_file[short_key].attrs["depth_scale"] = 1000.0

    env.env.load_observation_space()
    env.playback_episode(episode_id=episode_id, record_data=True)
    env.close_hdf5()

    log.info(f"Finished processing {final_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 demonstration files")
    parser.add_argument("--data_folder", type=str, default="data/behavior-1k/2025-challenge-rawdata", help="Path to the data folder")
    parser.add_argument("--task_id", type=int, default=21, help="Task ID to replay")
    parser.add_argument("--demo_id", type=int, default=210020, help="Demo ID to replay")
    parser.add_argument("--sampling_rate", type=int, default=1, help="Sampling rate for data extraction")
    parser.add_argument("--output_folder", type=str, default="./4d_latent_mapping/demonstration_replay", help="Output folder")
    parser.add_argument("--flush_threshold", type=int, default=50, help="Number of steps before flushing buffer to disk. Increase this if you have plenty of RAM to reduce I/O overhead.")
    parser.add_argument("--n_render_iterations", type=int, default=5, help="Number of render iterations per step for Omniverse raytracing stabilization.")
    parser.add_argument("--cameras", nargs='+', default=["head", "left_wrist", "right_wrist"], help="Cameras to replay")
    parser.add_argument("--instance_id_map", type=str, default=None, help="Path to fixed instance ID mapping JSON file. If not specified, uses /mnt/4d_latent_mapping/instance_id_map/task-{task_id:04d}/instance_id_map.json")

    args = parser.parse_args()
    
    # Load fixed instance ID mapping
    if args.instance_id_map is None:
        args.instance_id_map = f"/mnt/4d_latent_mapping/instance_id_map/task-{args.task_id:04d}/instance_id_map.json"

    fixed_instance_id_map = None
    if os.path.exists(args.instance_id_map):
        with open(args.instance_id_map, "r") as f:
            fixed_instance_id_map = json.load(f)
        log.info(f"Loaded fixed instance ID mapping with {len(fixed_instance_id_map)} entries from {args.instance_id_map}")
    else:
        log.warning(f"Instance ID mapping file not found: {args.instance_id_map}")

    all_camera_names = ROBOT_CAMERA_NAMES["R1Pro"]
    selected_camera_names = {key: val for key, val in all_camera_names.items() if key in args.cameras}
    if not selected_camera_names:
        selected_camera_names = all_camera_names

    task_name = TASK_INDICES_TO_NAMES[args.task_id]
    input_dir = os.path.join(args.data_folder, f"task-{args.task_id:04d}")
    
    if args.demo_id is not None:
        demo_ids = [args.demo_id]
    else:
        demo_files = sorted(glob.glob(os.path.join(input_dir, "episode_*.hdf5")))
        demo_ids = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in demo_files]

    task_scene_file_folder = os.path.join(
        os.path.dirname(os.path.dirname(og.__path__[0])), "joylo", "sampled_task", task_name
    )
    full_scene_file = None
    if os.path.exists(task_scene_file_folder):
        for file in os.listdir(task_scene_file_folder):
            if file.endswith(".json") and "partial_rooms" not in file:
                full_scene_file = os.path.join(task_scene_file_folder, file)
                break
    
    if not full_scene_file:
        log.error(f"No full scene file found for task {task_name}")
        return

    # Create env with smallest camera resolution, then upsize larger cameras in process_episode.
    # IsaacSim's VisionSensor supports upsizing resolution after creation (BEHAVIOR-1K pattern)
    # but downsizing can crash the render graph annotator detach.
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "depth_linear", "seg_instance_id"],
            "sensor_kwargs": {
                "image_height": MIN_IMAGE_SIZE,
                "image_width": MIN_IMAGE_SIZE,
            },
        },
    }

    env_wrapper = None
    real_env = None

    for demo_id in demo_ids:
        final_output_path = os.path.join(args.output_folder, f"task-{args.task_id:04d}", f"episode_{demo_id:08d}.hdf5")
        
        # Check if we should skip
        if os.path.exists(final_output_path):
            log.info(f"Skipping episode {demo_id} - output already exists.")
            continue

        input_path = os.path.join(input_dir, f"episode_{demo_id:08d}.hdf5")
        # We assume input_path is already local (handled by replay_task.py) so we don't copy it to tmp again
        local_input_path = input_path 
        dummy_output = f"./tmp/dummy_output_{demo_id}.hdf5"

        try:
            log.info(f"Processing {local_input_path}")

            if env_wrapper:
                if env_wrapper.input_hdf5:
                    env_wrapper.input_hdf5.close()
                if env_wrapper.hdf5_file:
                    env_wrapper.hdf5_file.close()

            if real_env is None:
                env_wrapper = BehaviorDataPlaybackWrapper.create_from_hdf5(
                    input_path=local_input_path,
                    output_path=dummy_output,
                    compression={"compression": "lzf"},
                    robot_obs_modalities=[],
                    robot_proprio_keys=[],
                    robot_sensor_config=robot_sensor_config,
                    n_render_iterations=args.n_render_iterations,
                    flush_every_n_traj=1,
                    flush_every_n_steps=args.flush_threshold,
                    include_robot_control=False,
                    include_contacts=False,
                    full_scene_file=full_scene_file,
                )
                real_env = env_wrapper.env
            else:
                 env_wrapper = BehaviorDataPlaybackWrapper(
                    env=real_env,
                    input_path=local_input_path,
                    output_path=dummy_output,
                    compression={"compression": "lzf"},
                    n_render_iterations=args.n_render_iterations,
                    flush_every_n_traj=1,
                    flush_every_n_steps=args.flush_threshold,
                    include_robot_control=False,
                    include_contacts=False,
                    full_scene_file=full_scene_file,
                )
            
            # Explicitly set the flush threshold for the wrapper's internal logic
            env_wrapper.flush_threshold = args.flush_threshold
            env_wrapper.fixed_instance_id_map = fixed_instance_id_map

            process_episode(
                env=env_wrapper,
                task_id=args.task_id,
                demo_id=demo_id,
                camera_names=selected_camera_names,
                sampling_rate=args.sampling_rate,
                output_folder=args.output_folder,
            )

        except Exception as e:
            log.error(f"Failed to process demo {demo_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(dummy_output):
                os.remove(dummy_output)

    og.shutdown()

if __name__ == "__main__":
    main()
