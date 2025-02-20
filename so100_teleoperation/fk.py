import numpy as np
import numpy.typing as npt
import time
# import matplotlib
# matplotlib.use("TkAgg")  # Spécifie un backend graphique interactif
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from feetech import Feetech
from reachy2_sdk import ReachySDK

from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
import threading
from pynput import keyboard

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


def plot_orientation(ax, position, orientation, scale=0.1):
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
    ax.plot(X, Y, Z, color='blue', linestyle='-', linewidth=2, label="Ligne")
    plot_orientation(ax, points[-1], orientation)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Graphe 3D avec 6 points reliés')
    ax.legend()
    ax.set_xlim([-0.2, 0.4])
    ax.set_ylim([-0.2, 0.4])
    ax.set_zlim([-0.2, 0.4])
    plt.draw()
    plt.pause(0.1)




# def show_robot_arm(points):

#     # Extraction des coordonnées X, Y, Z
#     X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

#     # Création de la figure 3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Tracé des points
#     ax.scatter(X, Y, Z, color='red', marker='o', s=50, label="Points")

#     # Tracé des lignes entre les points
#     ax.plot(X, Y, Z, color='blue', linestyle='-', linewidth=2, label="Ligne")

#     # Labels des axes
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Graphe 3D avec 6 points reliés')
#     ax.legend()

#     # Fixer l'échelle des axes entre -0.2 et 0.6
#     ax.set_xlim([-0.2, 0.6])
#     ax.set_ylim([-0.2, 0.6])
#     ax.set_zlim([-0.2, 0.6])

#     # Affichage du graphe
#     plt.show()


def fk(joints, ax, arm):
    joints = np.deg2rad(joints)

    p1 = np.array([0, 0, 0])
    p2 = np.array([0.031, 0. , 0.072])
    d3 = np.array([0.03, 0., 0.115])
    d4 = np.array([0.135, 0., 0.005])
    d5 = np.array([0.07, 0., 0.])

    # l1 = np.linalg.norm(d1)
    # l2 = np.linalg.norm(d2)
    # l3 = np.linalg.norm(d3)
    # l4 = np.linalg.norm(d4)

    
    # alpha = np.arctan2(d2[0], d2[2])
    # # print(alpha)

    # j2 = joints[1] + alpha
    # d2 = np.array([l2 * np.sin(j2), 0, l2 * np.cos(j2)])
    # # print(l2 * np.sin(j2), l2 * np.cos(j2))

    # beta = np.arctan2(d3[2], d3[0])
    # # print(beta)
    # j3 = -joints[2] - beta
    # d3 = np.array([l3 * np.cos(j3), 0, l3 * np.sin(j3)])

    # j4 = -joints[3]
    # d4 = np.array([l4 * np.cos(j4), 0, l4 * np.sin(j4)])

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
    P5 = T @ np.append(d5, 1)
    M5 = R.from_euler('xyz', [-joints[4], 0, 0], degrees=False).as_matrix()
    T5 = make_homogenous_matrix_from_rotation_matrix(M5, d5)
    T = T @ T5
    points = np.array([P1, P2[:3], P3[:3], P4[:3], P5[:3]])
    orientation = T[:3, :3]

    pose = make_homogenous_matrix_from_rotation_matrix(orientation, P5[:3])
    # return pose
    



    # P2 = d1
    # P3 = d1 + d2
    # P4 = d1 + d2 + d3
    # P5 = d1 + d2 + d3 + d4

    # m1 = R.from_euler('xyz', [0, 0, -joints[0]], degrees=False)
    # m2 = R.from_euler('xyz', [0, joints[1], 0], degrees=False)
    # m3 = R.from_euler('xyz', [0, -joints[2], 0], degrees=False)
    # m4 = R.from_euler('xyz', [0, 0, -joints[3]], degrees=False)
    # m5 = R.from_euler('xyz', [joints[4], 0, 0], degrees=False)

    # rot = m1
    # P2 = rot.apply(P2)
    # rot = rot * m2
    # P3 = rot.apply(P3)
    # rot = rot * m3
    # P4 = rot.apply(P4)
    # rot = rot * m4
    # P5 = rot.apply(P5)


    # rotation = R.from_euler('xyz', [0, 0, -joints[0]], degrees=False)
    # P2 = rotation.apply(P2)
    # P3 = rotation.apply(P3)
    # P4 = rotation.apply(P4)
    # P5 = rotation.apply(P5)


    # if arm == "l_arm":
    #     print(P5)
    #     P5[1] *= -1
    #     orientation = R.from_matrix(orientation).as_euler('xyz', degrees=False)
    #     orientation[0] = -orientation[0]
    #     orientation[2] = -orientation[2]
    #     orientation = R.from_euler('xyz', orientation, degrees=False).as_matrix()
    #     pose = make_homogenous_matrix_from_rotation_matrix(orientation, P5[:3])
    #     print(P5)
    #     print(pose)
 
 
    
    # orientation = m1 * m2 * m3 * m4 * m5

    # points = np.array([P1, P2, P3, P4, P5])
    # print(points)

    
    # update_plot(ax, points, orientation)
    return pose


def gripper_control(joint, arm):
    min_joint, max_joint = -60, -15
    min_gripper, max_gripper = 0, 130
    gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper

    if arm == "r_arm":
        reachy.r_arm.gripper.goal_position = gripper_opening
        reachy.r_arm.gripper.send_goal_positions()
    elif arm == "l_arm":
        reachy.l_arm.gripper.goal_position = gripper_opening
        reachy.l_arm.gripper.send_goal_positions()

def antena_control(joint):
    min_joint, max_joint = -60, 0
    min_antenna, max_antenna = 30, -160
    if joint < -60:
        joint = -60
    if joint > -0:
        joint = -0
    antenna_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_antenna - min_antenna) + min_antenna

    reachy.head.r_antenna.goal_position = antenna_opening
    reachy.head.l_antenna.goal_position = -antenna_opening
    reachy.send_goal_positions()


def init_gripper(joint):

    #TODO
    min_joint, max_joint = -60, -15
    min_gripper, max_gripper = 0, 130
    gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper


    for i in range(100):

        reachy.r_arm.gripper.goal_position = gripper_opening
        reachy.r_arm.gripper.send_goal_positions()


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    if arm == "r_arm":
        request = ArmCartesianGoal(
            id=reachy.r_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.UNFREEZE,
            constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.05),
            order_id=Int32Value(value=5),
        )
        reachy.r_arm._stub.SendArmCartesianGoal(request)
        # print(reachy.r_arm.shoulder.pitch.present_position)
    elif arm == "l_arm":
        request = ArmCartesianGoal(
            id=reachy.l_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.UNFREEZE,
            constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.05),
            order_id=Int32Value(value=5),
        )
        reachy.l_arm._stub.SendArmCartesianGoal(request)

def on_press(key):
    global mode
    global ignore
    global position_coeff
    global orientation_coeff
    global arm
    print(f'Key {key} pressed')
    try:
        if key.char == 'a':
            mode = 1
        elif key.char == 'b':
            mode = 2
        elif key.char == 'c':
            mode = 3
        elif key.char == 'd':
            ignore = not ignore
        elif key.char == 'e':
            mode = 4
        elif key.char == 'p':
            position_coeff += 0.1
            print(f'Position coeff: {position_coeff}')
        elif key.char == 'm':
            position_coeff -= 0.1
            print(f'Position coeff: {position_coeff}')
        elif key.char == 'o':
            orientation_coeff += 0.1
            print(f'Orientation coeff: {orientation_coeff}')
        elif key.char == 'l':
            orientation_coeff -= 0.1
            print(f'Orientation coeff: {orientation_coeff}')
        elif key.char == 'r':
            arm = "r_arm"
        elif key.char == 't':
            arm = "l_arm"
        elif key.char == 'h':
            mode = 5
        
        print(f'Mode actuel: {mode}')
    except AttributeError:
        print(f'Key {key} pressed')
        pass

def listen_keyboard():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()



def find_reachy_pose(so_pose, previous_reachy_pose, previous_so_pose):

    # print(so_pose)
    # print(previous_so_pose)
    # print(previous_reachy_pose)
    reachy_pose = previous_reachy_pose.copy()
    # print(so_pose[:3, 3])
    diff_position = so_pose[:3, 3] - previous_so_pose[:3, 3]
    reachy_pose[:3, 3] += diff_position

    diff_orientation = so_pose[:3, :3] @ previous_so_pose[:3, :3].T
    reachy_pose[:3, :3] = diff_orientation @ previous_reachy_pose[:3, :3]

    return reachy_pose


def find_reachy_pose2(so_pose, mode, arm):
    if mode == 1:
        reachy_orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False).as_matrix()
    if mode == 2:
        reachy_orientation = R.from_euler('xyz', [0, 0, np.pi/4], degrees=False).as_matrix()

    # reachy_orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False).as_matrix()
    
    reachy_position = [0.36, -0.2, -0.28]
    if arm == "l_arm":
        reachy_position = [0.36, 0.2, -0.28]
    reachy_pose_base = make_homogenous_matrix_from_rotation_matrix(reachy_orientation, reachy_position)

    so_pose_base = np.array([[-0.04154232, -0.01284393,  0.99905419,  0.15860088],
    [-0.03067243,  0.99946248,  0.01157377,  0.11710165],
    [-0.99866583, -0.03016262, -0.04191395,  0.07640968],
    [ 0.,          0.,          0. ,         1. ,       ]])

    diff_position = so_pose[:3, 3] - so_pose_base[:3, 3]
    reachy_pose = reachy_pose_base.copy()
    reachy_pose[:3, 3] += diff_position

    diff_orientation = so_pose[:3, :3] @ so_pose_base[:3, :3].T
    reachy_pose[:3, :3] = diff_orientation @ reachy_pose_base[:3, :3]
    return reachy_pose


def find_reachy_pose3(so_pose, previous_reachy_pose, previous_so_pose):
    global position_coeff
    global orientation_coeff

    reachy_pose = previous_reachy_pose.copy()
    diff_position = so_pose[:3, 3] - previous_so_pose[:3, 3]
    diff_position *= position_coeff
    reachy_pose[:3, 3] += diff_position

    diff_orientation = so_pose[:3, :3] @ previous_so_pose[:3, :3].T
    diff_orientation = R.from_matrix(diff_orientation).as_euler('xyz', degrees=False)
    diff_orientation *= orientation_coeff
    diff_orientation = R.from_euler('xyz', diff_orientation, degrees=False).as_matrix()
    reachy_pose[:3, :3] = diff_orientation @ previous_reachy_pose[:3, :3]

    return reachy_pose


def control_head(pose):
    print(f" 4 : {pose[4]}")
    print(f" 0 : {pose[0]}")
    print(f" 1 : {pose[1]}")
    reachy_roll_range = [-40, 40]
    reachy_pitch_range = [-40, 40]
    so_roll_range = [-100, 0]
    so_pitch_range = [-40, 40]

    head_roll = np.interp(pose[0], so_roll_range, reachy_roll_range)
    head_pitch = np.interp(pose[1], so_pitch_range, reachy_pitch_range)
    head_yaw = pose[4] + 65

    reachy.head.neck.roll.goal_position = head_roll
    reachy.head.neck.pitch.goal_position = head_pitch
    reachy.head.neck.yaw.goal_position = head_yaw
    reachy.head.send_goal_positions()


if __name__  == "__main__":
    fig, ax = create_plot()
    feetech = Feetech("/dev/ttyACM0")
    time.sleep(1)
    reachy = ReachySDK(host="localhost")
    reachy.turn_on()
    reachy.head.l_antenna.turn_on()
    reachy.head.r_antenna.turn_on()

    position = [0.36, -0.2, -0.28]
    orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
    pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
    reachy.r_arm.send_cartesian_interpolation(pose, 2)
    position = [0.36, 0.2, -0.28]
    orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
    l_pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
    reachy.l_arm.send_cartesian_interpolation(l_pose, 2)

    frequency = 100
    previous_reachy_pose = pose

    mode = 1
    # Démarrer l'écouteur dans un thread séparé
    keyboard_thread = threading.Thread(target=listen_keyboard, daemon=True)
    keyboard_thread.start()

    position_coeff = 1
    orientation_coeff = 1
    arm = "r_arm"


    pos = []
    for i in range(1,7):
        position = feetech.get_position(i)
        pos.append(position)
    pose = fk(pos, ax, arm)
    previous_so_pose = pose 

    ignore = False

    while True:
        t = time.time()
        pos = []
        for i in range(1,7):
            position = feetech.get_position(i)
            pos.append(position)
        
        if mode == 1 or mode == 2 or mode == 3 or mode == 4:
            gripper_control(pos[5], arm)
        if mode == 5:
            antena_control(pos[5])
            control_head(pos)

        # pos = [0, 0, 0., 0, 0, 0]
        so_pose = fk(pos, ax, arm)
        if mode == 1 or mode == 2:
            reachy_pose = find_reachy_pose2(so_pose,mode, arm)
        elif mode == 3:
            if not ignore:
                reachy_pose = find_reachy_pose(so_pose, previous_reachy_pose, previous_so_pose)
        elif mode == 4:
            reachy_pose = find_reachy_pose3(so_pose, previous_reachy_pose, previous_so_pose)
        go_to_pose(reachy, reachy_pose, arm)
        # go_to_pose(reachy, l_pose, "l_arm")
        previous_reachy_pose = reachy_pose
        previous_so_pose = so_pose
        time.sleep(max(1/frequency - (time.time() - t), 0))


    
    # show_robot_arm(points)
    



