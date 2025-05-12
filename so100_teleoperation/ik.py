from utils import create_plot, make_homogenous_matrix_from_rotation_matrix, fk
from feetech import Feetech
from scipy.spatial.transform import Rotation as R  # type: ignore
import numpy as np
import time


def ik(pose, ax=None):
    d1 = np.array([0, 0, 0])
    d2 = np.array([0.031, 0.0, 0.072])
    d3 = np.array([0.03, 0.0, 0.115])
    d4 = np.array([0.135, 0.0, 0.005])
    d5 = np.array([0.07, 0.0, 0.0])

    joints = [0, 0, 0, 0, 0]

    goal_position = pose[:3, 3]
    joints[0] = np.arctan2(-goal_position[1], goal_position[0])

    rotation = R.from_euler("xyz", [0, 0, joints[0]], degrees=False).as_matrix()
    position = np.array([0, 0, 0])
    T_base = make_homogenous_matrix_from_rotation_matrix(rotation, position)
    T_goal = np.dot(T_base, pose)
    T_goal_orientation = T_goal[:3, :3]
    T_goal_orientation = R.from_matrix(T_goal_orientation).as_euler(
        "xyz", degrees=False
    )
    p4 = np.dot(T_goal, [-d5[0], 0, 0, 1])[:3]
    p = p4 - d2
    alpha = np.arctan2(d3[0], d3[2])
    b = np.linalg.norm(d3)
    a = np.linalg.norm(d4)
    c = np.linalg.norm(p4 - d2)
    cosA = (b**2 + c**2 - a**2) / (2 * b * c)
    cosA = max(-1, min(1, cosA))
    beta = np.arccos(cosA)
    gamma = np.arctan2(p[2], p[0])
    joints[1] = np.pi / 2 - (alpha + beta + gamma)

    rotation1 = R.from_euler("xyz", [0, 0, -joints[0]], degrees=False).as_matrix()
    rotation2 = R.from_euler("xyz", [0, joints[1], 0], degrees=False).as_matrix()
    T_2_1 = make_homogenous_matrix_from_rotation_matrix(rotation2, d2)
    p4 = [p4[0], p4[1], p4[2], 1]
    p4_2 = np.dot(np.linalg.inv(T_2_1), p4)
    p4 = p4_2[:3] - d3
    alpha = np.arctan2(d4[2], d4[0])
    beta = np.arctan2(p4[2], p4[0])
    joints[2] = alpha - beta

    rotation3 = R.from_euler("xyz", [0, joints[2], 0], degrees=False).as_matrix()
    T_3_2 = make_homogenous_matrix_from_rotation_matrix(rotation3, d3)
    p4 = np.dot(np.linalg.inv(T_3_2), p4_2)[:3]
    P_goal_2 = np.dot(np.linalg.inv(T_2_1), T_goal[:4, 3])
    P_goal_3 = np.dot(np.linalg.inv(T_3_2), P_goal_2)
    p_goal_4 = P_goal_3[:3] - p4

    alpha = np.arctan2(p_goal_4[2], p_goal_4[0])
    joints[3] = -alpha

    rotation1 = R.from_euler("xyz", [0, 0, -joints[0]], degrees=False).as_matrix()
    rotation2 = R.from_euler("xyz", [0, joints[1], 0], degrees=False).as_matrix()
    rotation3 = R.from_euler("xyz", [0, joints[2], 0], degrees=False).as_matrix()
    rotation4 = R.from_euler("xyz", [0, joints[3], 0], degrees=False).as_matrix()
    rotation = rotation1 @ rotation2 @ rotation3 @ rotation4
    T_4_base = make_homogenous_matrix_from_rotation_matrix(rotation, goal_position)
    T_goal_base = np.dot(np.linalg.inv(T_4_base), pose)
    rot_euler = R.from_matrix(T_goal_base[:3, :3]).as_euler("xyz", degrees=False)
    joints[4] = -rot_euler[0]

    joints = np.rad2deg(joints)

    return joints


if __name__ == "__main__":
    # Test
    feetech = Feetech("/dev/ttyACM0")
    fig, ax = create_plot()
    ax = None

    while True:
        pose = np.array(
            [
                [0.0268145, 0.12638273, 0.99161907, 0.24835922],
                [0.03061745, 0.99140653, -0.12718357, 0.05155435],
                [-0.99917143, 0.03377121, 0.02271455, 0.12995589],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        joints2 = ik(pose, ax=ax)
        print(joints2)

        pose2 = fk(joints2)
        is_working = np.allclose(pose, pose2, atol=0.001)
        print(is_working)
        print("______________")
        feetech.goto_joints(joints2, duration=3)
        time.sleep(2)
