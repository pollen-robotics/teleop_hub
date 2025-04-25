import copy

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def normalize_vector(v):
    """Normalize a vector and handle the case where the norm is zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def rotation_matrix_from_vector(vect: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix aligning [0, 1, 0] to the given vect."""
    vect1 = np.array([0, 1, 0])
    eps = 1e-6  # tolerance for numerical stability

    # Normalize the input vector
    vect = vect / (np.linalg.norm(vect) + eps)

    # Special case: aligned or opposite vector
    if np.allclose(vect1, vect, atol=eps):
        return np.eye(3)
    if np.allclose(vect1, -vect, atol=eps):
        return np.diag([-1, -1, 1])

    # Compute the rotation axis and angle
    rotation_vector = np.cross(vect1, vect)
    sin_theta = np.linalg.norm(rotation_vector)
    cos_theta = np.dot(vect1, vect)

    # Construction of the rotation matrix using Rodrigues' rotation formula
    if sin_theta > eps:
        axis = rotation_vector / sin_theta
        angle = np.arctan2(sin_theta, cos_theta)
        rotation_matrix = R.from_rotvec(axis * angle).as_matrix()
    else:
        # Handle the case where sin_theta is very small (near gimbal lock)
        rotation_matrix = np.eye(3)

    return rotation_matrix


def make_homogenous_matrix_from_rotation_matrix(rotation_matrix, position):
    """Convert a 3x3 rotation matrix to a 4x4 homogenous matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix


def limit_orbita3d_joints(joints: list[float], orbita3D_max_angle: float) -> list[float]:
    """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D,
    i.e. casting into Orbita's cone.
    """
    joints = copy.deepcopy(joints)
    rotation = R.from_euler("XYZ", [joints[0], joints[1], joints[2]], degrees=False)
    new_joints = rotation.as_euler("ZYZ", degrees=False)
    new_joints[1] = min(orbita3D_max_angle, max(-orbita3D_max_angle, new_joints[1]))
    rotation = R.from_euler("ZYZ", new_joints, degrees=False)
    [roll, pitch, yaw] = rotation.as_euler("XYZ", degrees=False)
    joints = [float(roll), float(pitch), float(yaw)]
    return joints
