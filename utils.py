import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


def normalize_vector(v):
    """Normalize a vector and handle the case where the norm is zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def rotation_matrix_from_vector(vect: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix aligning [0, 0, -1] to the given vect."""
    vect1 = np.array([0, 0, -1])
    eps = 1e-6  # tolerance for numerical stability

    vect = vect / (np.linalg.norm(vect) + eps)

    # Special case : aligned or opposite vector
    if np.allclose(vect1, vect, atol=eps):
        return np.eye(3)
    if np.allclose(vect1, -vect, atol=eps):
        return np.diag([-1, -1, 1])

    # Compute the rotation axis and angle
    rotation_vector = np.cross(vect1, vect)
    sin_theta = np.linalg.norm(rotation_vector)
    cos_theta = np.dot(vect1, vect)

    # Construction of the rotation matrix
    axis = rotation_vector / (sin_theta + eps)
    angle = np.arctan2(sin_theta, cos_theta)
    rotation_matrix = R.from_rotvec(axis * angle).as_matrix()

    return rotation_matrix
