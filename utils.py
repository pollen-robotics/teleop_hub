import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import time

from scipy.spatial.transform import Rotation as R


def make_homogenous_matrix_from_rotation_matrix(rotation_matrix, position):
    """Convert a 3x3 rotation matrix to a 4x4 homogenous matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix

def create_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def plot_orientation(ax, position, orientation, scale=0.4):
    """
    Ajoute un repère d'orientation au dernier point en utilisant roll, pitch et yaw.
    """
    # rotation_matrix = R.from_euler('xyz', orientation, degrees=False).as_matrix()
    origin = position
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    # print(orientation)
    for i in range(3):
        direction = orientation[:, i] * scale
        ax.quiver(*origin, *direction, color=colors[i], label=f"{labels[i]}-axis")

def update_plot(ax, points, orientation):
    # print(points)
    ax.clear()
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(X, Y, Z, color='red', marker='o', s=50, label="Points")
    ax.plot(X, Y, Z, color='blue', linestyle='-', linewidth=2, label="Ligne", alpha=0.5)
    plot_orientation(ax, points[-1], orientation)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Graphe 3D avec 6 points reliés')
    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.draw()
    plt.pause(0.001)

def fk(joints, top_grasp = False, ax=None):
    joints = np.deg2rad(joints)

    p1 = np.array([0, 0, 0])
    p2 = np.array([0.031, 0. , 0.072])
    d3 = np.array([0.03, 0., 0.115])
    d4 = np.array([0.135, 0., 0.005])
    d4_bis = np.array([0.06, 0, 0])
    d5 = np.array([0.07, 0., 0.])

    P1 = p1
    M1 = R.from_euler('xyz', [0, 0, -joints[0]], degrees=False).as_matrix()
    T1 = make_homogenous_matrix_from_rotation_matrix(M1, p1)
    T = T1
    P2 = T @ np.append(p2, 1)
    M2 = R.from_euler('xyz', [0, joints[1], 0], degrees=False).as_matrix()
    T2 = make_homogenous_matrix_from_rotation_matrix(M2, p2)
    T = T @ T2
    P3 = T @ np.append(d3, 1)
    M3 = R.from_euler('xyz', [0, joints[2], 0], degrees=False).as_matrix()
    T3 = make_homogenous_matrix_from_rotation_matrix(M3, d3)
    T = T @ T3
    P4 = T @ np.append(d4, 1)
    M4 = R.from_euler('xyz', [0, joints[3], 0], degrees=False).as_matrix()
    T4 = make_homogenous_matrix_from_rotation_matrix(M4, d4)
    T = T @ T4
    P4_bis = T @ np.append(d4_bis, 1)
    M4_bis = R.from_euler('xyz', [0, 0, joints[6]], degrees=False).as_matrix()
    T4_bis = make_homogenous_matrix_from_rotation_matrix(M4_bis, d4_bis)
    T = T @ T4_bis
    P5 = T @ np.append(d5, 1)
    M5 = R.from_euler('xyz', [-joints[4], 0, 0], degrees=False).as_matrix()
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