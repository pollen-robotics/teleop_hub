import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from scipy.spatial.transform import Rotation as R  # type: ignore


def make_homogenous_matrix_from_rotation_matrix(rotation_matrix, position):
    """Convert a 3x3 rotation matrix to a 4x4 homogenous matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix


def create_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def plot_orientation(ax, position, orientation, scale=0.1):
    """
    Ajoute un repère d'orientation au dernier point en utilisant roll, pitch et yaw.
    """
    origin = position
    colors = ["r", "g", "b"]
    labels = ["X", "Y", "Z"]

    for i in range(3):
        direction = orientation[:, i] * scale
        ax.quiver(*origin, *direction, color=colors[i], label=f"{labels[i]}-axis")


def update_plot(ax, points, orientation):
    ax.clear()
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(X, Y, Z, color="red", marker="o", s=50, label="Points")
    ax.plot(X, Y, Z, color="blue", linestyle="-", linewidth=2, label="Ligne", alpha=0.5)
    plot_orientation(ax, points[-1], orientation)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Graphe 3D avec 6 points reliés")
    ax.legend()
    ax.set_xlim([-0.2, 0.4])
    ax.set_ylim([-0.2, 0.4])
    ax.set_zlim([-0.2, 0.4])
    plt.draw()
    plt.pause(0.1)


def fk(joints, top_grasp=False, ax=None):
    joints = np.deg2rad(joints)

    p1 = np.array([0, 0, 0])
    p2 = np.array([0.031, 0.0, 0.072])
    d3 = np.array([0.03, 0.0, 0.115])
    d4 = np.array([0.135, 0.0, 0.005])
    d4_bis = np.array([0.06, 0, 0])
    d5 = np.array([0.07, 0.0, 0.0])

    P1 = p1
    M1 = R.from_euler("xyz", [0, 0, -joints[0]], degrees=False).as_matrix()
    T1 = make_homogenous_matrix_from_rotation_matrix(M1, p1)
    T = T1
    P2 = T @ np.append(p2, 1)
    M2 = R.from_euler("xyz", [0, joints[1], 0], degrees=False).as_matrix()
    T2 = make_homogenous_matrix_from_rotation_matrix(M2, p2)
    T = T @ T2
    P3 = T @ np.append(d3, 1)
    M3 = R.from_euler("xyz", [0, joints[2], 0], degrees=False).as_matrix()
    T3 = make_homogenous_matrix_from_rotation_matrix(M3, d3)
    T = T @ T3
    P4 = T @ np.append(d4, 1)
    M4 = R.from_euler("xyz", [0, joints[3], 0], degrees=False).as_matrix()
    T4 = make_homogenous_matrix_from_rotation_matrix(M4, d4)
    T = T @ T4
    if len(joints) == 7:
        P4_bis = T @ np.append(d4_bis, 1)
        M4_bis = R.from_euler("xyz", [0, 0, joints[6]], degrees=False).as_matrix()
        T4_bis = make_homogenous_matrix_from_rotation_matrix(M4_bis, d4_bis)
        T = T @ T4_bis
    P5 = T @ np.append(d5, 1)
    M5 = R.from_euler("xyz", [-joints[4], 0, 0], degrees=False).as_matrix()
    T5 = make_homogenous_matrix_from_rotation_matrix(M5, d5)
    T = T @ T5
    points = np.array([P1, P2[:3], P3[:3], P4[:3], P5[:3]])
    orientation = T[:3, :3]
    # rot1 = R.from_euler('xyz', [0, 0, -np.pi/4], degrees=False).as_matrix()
    # rot2 = R.from_euler('xyz', [0, np.pi/2, 0], degrees=False).as_matrix()
    # if top_grasp:
    #     orientation = orientation @ rot2 @ rot1

    pose = make_homogenous_matrix_from_rotation_matrix(orientation, P5[:3])

    if ax is not None:
        update_plot(ax, points, orientation)
    return pose


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

    T_base
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
