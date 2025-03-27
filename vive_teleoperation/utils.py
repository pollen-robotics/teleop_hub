import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

def make_homogenous_matrix_from_rotation_matrix(rotation_matrix, position):
    """Convert a 3x3 rotation matrix to a 4x4 homogenous matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix

def limit_orbita3d_joints(joints: list[float], orbita3D_max_angle: float) -> list[float]:
    """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D. i.e. casting into Orbita's cone."""
    joints = copy.deepcopy(joints)
    rotation = R.from_euler("XYZ", [joints[0], joints[1], joints[2]], degrees=False)
    new_joints = rotation.as_euler("ZYZ", degrees=False)
    new_joints[1] = min(orbita3D_max_angle, max(-orbita3D_max_angle, new_joints[1]))
    rotation = R.from_euler("ZYZ", new_joints, degrees=False)
    [roll, pitch, yaw] = rotation.as_euler("XYZ", degrees=False)
    joints = [float(roll), float(pitch), float(yaw)]
    return joints