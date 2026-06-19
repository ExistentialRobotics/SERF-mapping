"""
Pose transformation utilities for converting between coordinate systems.

OmniGibson uses a right-handed coordinate system with Z-up.
OpenCV uses a right-handed coordinate system with Y-down, Z-forward.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple

__all__ = [
    "pose_to_extrinsic_cv",
    "extrinsic_cv_to_pose",
    "look_at_extrinsic_cv",
]

# Transformation matrix from OmniGibson to OpenCV coordinate system
_T_CV_OG = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


def pose_to_extrinsic_cv(pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
    """
    Convert position and orientation to extrinsic matrix in OpenCV format (world-to-camera).

    Args:
        pos: 3D position in world coordinates
        orn: Quaternion in scipy format [x, y, z, w]

    Returns:
        4x4 extrinsic matrix in OpenCV convention (world-to-camera)
    """
    R_world_og = Rotation.from_quat(orn).as_matrix()
    t_world_og = np.asarray(pos)

    R_og_world = R_world_og.T
    t_og_world = -R_og_world @ t_world_og

    T_og_world = np.eye(4)
    T_og_world[:3, :3] = R_og_world
    T_og_world[:3, 3] = t_og_world

    extrinsic_cv = _T_CV_OG @ T_og_world
    return extrinsic_cv.astype(np.float32)


def extrinsic_cv_to_pose(extrinsic_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert extrinsic matrix in OpenCV format back to position and orientation.

    Args:
        extrinsic_cv: 4x4 extrinsic matrix in OpenCV convention (world-to-camera)

    Returns:
        pos: 3D position in world coordinates
        orn: Quaternion in scipy format [x, y, z, w]
    """
    T_og_world = _T_CV_OG @ extrinsic_cv

    R_og_world = T_og_world[:3, :3]
    t_og_world = T_og_world[:3, 3]

    R_world_og = R_og_world.T
    t_world_og = -R_world_og @ t_og_world

    pos = t_world_og
    orn = Rotation.from_matrix(R_world_og).as_quat()

    return pos, orn


def look_at_extrinsic_cv(
    cam_pos: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Compute 4x4 extrinsic CV matrix for a camera looking at a target.

    Builds a camera frame (X=right, Y=up, -Z=forward in OmniGibson convention)
    and converts to OpenCV world-to-camera extrinsic via ``pose_to_extrinsic_cv``.

    Args:
        cam_pos: (3,) camera position in world coordinates.
        target: (3,) point the camera looks at.
        world_up: (3,) world up direction (default: Z-up).

    Returns:
        (4, 4) float32 extrinsic matrix in OpenCV convention.
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera is looking straight up or down — use Y as fallback up
        world_up_fallback = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up_fallback)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    up = np.cross(right, forward)

    # OmniGibson camera: X=right, Y=up, -Z=forward
    R_world_cam = np.column_stack([right, up, -forward])
    orn = Rotation.from_matrix(R_world_cam).as_quat()  # (x, y, z, w)

    return pose_to_extrinsic_cv(cam_pos, orn)
