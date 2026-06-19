import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.geometry import (
    calculate_intrinsics,
    unproject_depth_to_world,
    solve_rigid_transform,
    refine_registration_icp,
)
from utils.optimizer import update_optimizer_param
from utils.visualization import (
    TorchPCA,
    run_pca_visualization,
    draw_geometries_with_key_callbacks,
    apply_camera_extrinsic,
    compute_pca_colors,
    build_link_colors,
)
from utils.pose import pose_to_extrinsic_cv, extrinsic_cv_to_pose, look_at_extrinsic_cv
from utils.instance_ids import (
    load_instance_id_map_from_source,
    load_instance_id_map_from_hdf5,
    load_instance_id_map_from_json,
    filter_ids_by_keywords,
    resolve_names_to_ids,
    build_cross_episode_id_mapping,
)
from utils.robot import (
    URDF_PATH,
    load_robot_states,
    state_to_urdf_cfg,
    joint_qpos_to_urdf_cfg,
    state_to_base_pose,
    RobotSurfaceSampler,
)

__all__ = [
    "calculate_intrinsics",
    "unproject_depth_to_world",
    "solve_rigid_transform",
    "refine_registration_icp",
    "update_optimizer_param",
    "TorchPCA",
    "run_pca_visualization",
    "draw_geometries_with_key_callbacks",
    "apply_camera_extrinsic",
    "compute_pca_colors",
    "build_link_colors",
    "pose_to_extrinsic_cv",
    "extrinsic_cv_to_pose",
    "look_at_extrinsic_cv",
    "load_instance_id_map_from_source",
    "load_instance_id_map_from_hdf5",
    "load_instance_id_map_from_json",
    "filter_ids_by_keywords",
    "resolve_names_to_ids",
    "build_cross_episode_id_mapping",
    "URDF_PATH",
    "load_robot_states",
    "state_to_urdf_cfg",
    "joint_qpos_to_urdf_cfg",
    "state_to_base_pose",
    "RobotSurfaceSampler",
]
