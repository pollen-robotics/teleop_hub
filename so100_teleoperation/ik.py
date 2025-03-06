from utils import create_plot, plot_orientation, make_homogenous_matrix_from_rotation_matrix, update_plot, fk
from feetech import Feetech
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

def ik(pose, ax=None):

    d1 = np.array([0, 0, 0])
    d2 = np.array([0.031, 0. , 0.072])
    d3 = np.array([0.03, 0., 0.115])
    d4 = np.array([0.135, 0., 0.005])
    d5 = np.array([0.07, 0., 0.])


    print(pose)
    goal_position = pose[:3, 3]
    print(goal_position)
    goal_orientation = pose[:3, :3]
    goal_orientation = R.from_matrix(goal_orientation).as_euler('xyz', degrees=False)
    print(goal_orientation)

    joints = [0, 0, 0, 0, 0]

    joints[0] = np.arctan2(-goal_position[1], goal_position[0])
    
    rotation = R.from_euler('xyz', [0, 0, joints[0]], degrees=False).as_matrix()
    position = np.array([0, 0, 0])
    T_base = make_homogenous_matrix_from_rotation_matrix(rotation, position)
    # T_base_inv = np.linalg.inv(T_base)
    T_goal = np.dot(T_base, [goal_position[0], goal_position[1], goal_position[2], 1])
    T_goal = np.dot(T_base, pose)
    T_goal_position = T_goal[:3]
    T_goal_orientation = T_goal[:3, :3]
    T_goal_orientation = R.from_matrix(T_goal_orientation).as_euler('xyz', degrees=False)
    # print(T_goal_orientation)
    # print(f"T_goal_position: {T_goal_position}")
    # print(f"T_goal_orientation: {T_goal_orientation}")
    # print(goal_orientation)
    # joints[4] = T_goal_orientation[0]
    # joints[4] = (- T_goal_orientation[0] + np.pi)
    # if joints[4] > np.pi:
    #     joints[4] -= 2*np.pi
    # print(f"T_goal: {T_goal}")

    # T_goal[:3, 3] += d2
    p4 = np.dot(T_goal, [-d5[0], 0, 0, 1])[:3]

    # print(p4 - d2)
    p  = p4 - d2
    alpha = np.arctan2(d3[0], d3[2])
    alpha2 = np.arctan2(d4[2], d4[0])
    # print(alpha)
    b = np.linalg.norm(d3)
    a = np.linalg.norm(d4)
    c = np.linalg.norm(p4 - d2)
    beta = np.arccos((b**2 + c**2 - a**2)/(2*b*c))
    gamma = np.arctan2(p[2], p[0])
    # print(f"gamma: {gamma}")
    # print(f"beta: {beta}")
    # print(f"alpha: {alpha}")
    print(alpha + beta + gamma)
    joints[1] = np.pi/2 - (alpha + beta + gamma)

    # alphaD = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    # joints[2] = alphaD - alpha + alpha2

    T_base

    rotation1 = R.from_euler('xyz', [0, 0, -joints[0]], degrees=False).as_matrix()
    rotation2 = R.from_euler('xyz', [0, joints[1], 0], degrees=False).as_matrix()
    T_2_1 = make_homogenous_matrix_from_rotation_matrix(rotation2, d2)
    # T_2_base = np.dot(T_base, T_2_1)
    # p4 = [1, 0, 0, 1]
    p4 = [p4[0], p4[1], p4[2], 1]
    p4_2 = np.dot(np.linalg.inv(T_2_1), p4)
    # print(f"T_2_base: {T_2_1}")
    # print(f"p4_2: {p4_2}")

    p4 = p4_2[:3] - d3

    alpha = np.arctan2(d4[2], d4[0])
    # print(f"alpha: {alpha}")

    beta = np.arctan2(p4[2], p4[0])
    joints[2] = alpha - beta

    # print(p4)

    rotation3 = R.from_euler('xyz', [0, joints[2], 0], degrees=False).as_matrix()
    T_3_2 = make_homogenous_matrix_from_rotation_matrix(rotation3, d3)

    p4 = np.dot(np.linalg.inv(T_3_2), p4_2)[:3]
    # p_goal_base = np.dot(np.linalg.inv(T_base), pose[:4, 3])
    # print(f"p_goal_base: {p_goal_base}")
    P_goal_2 = np.dot(np.linalg.inv(T_2_1), T_goal[:4, 3])
    # print(f"P_goal_2: {P_goal_2}")
    P_goal_3 = np.dot(np.linalg.inv(T_3_2), P_goal_2)

    p_goal_4 = P_goal_3[:3] - p4
    # print(f"p_goal_4: {p_goal_4}")

    alpha = np.arctan2(p_goal_4[2], p_goal_4[0])
    joints[3] = -alpha



    rotation1 = R.from_euler('xyz', [0, 0, -joints[0]], degrees=False).as_matrix()
    rotation2 = R.from_euler('xyz', [0, joints[1], 0], degrees=False).as_matrix()
    rotation3 = R.from_euler('xyz', [0, -joints[2], 0], degrees=False).as_matrix()
    rotation4 = R.from_euler('xyz', [0, joints[3], 0], degrees=False).as_matrix()

    rotation = rotation1 @ rotation2 @ rotation3 @ rotation4
    # print(f"rotation: {rotation}")
    # rotation = R.from_matrix(rotation).as_euler('xyz', degrees=False)
    # print(f"rotation: {np.rad2deg(rotation)}")

    px = [0, 0.1, 0, 1]

    T_4_base = make_homogenous_matrix_from_rotation_matrix(rotation, goal_position)
    px1 = np.dot(T_4_base, px)[:3]
    px2 = np.dot(pose, px)[:3]
    print(f"px: {px1}")
    # px1 = px1 - goal_position

    px1 = px1 - goal_position
    px2 = px2 - goal_position
    px3 = px2 - px1
    angle1 = np.arctan2(px3[1], px3[0])
    angle2 = np.arctan2(px3[0], px3[1])
    angle3 = np.arctan2(-px3[1], px3[0])
    angle4 = np.arctan2(-px3[0], px3[1])
    print(f"angle1: {np.rad2deg(angle1)}")
    print(f"angle2: {np.rad2deg(angle2)}")
    print(f"angle3: {np.rad2deg(angle3)}")
    print(f"angle4: {np.rad2deg(angle4)}")
    
    
    # px1 = np.dot(np.linalg.inv(rotation1), px1)
    # px2 = np.dot(np.linalg.inv(rotation1), px2)

    ax.plot(px1[0], px1[1], px1[2], 'go')
    ax.plot(px2[0], px2[1], px2[2], 'bo')
    ax.plot(px3[0], px3[1], px3[2], 'ro')
    plt.draw()
    plt.pause(0.1)

    print(f"px1: {px1}")
    print(f"px2: {px2}")

  
    
    # rot = np.dot(rotation, np.linalg.inv(pose[:3, :3]))
    # rot = R.from_matrix(rot).as_euler('xyz', degrees=False)
    # print(rot
    # p6 = np.dot(T_goal, [-d5[0], -d4[2], 0, 1])[:3]
    # p7 = [d2[0] + d3[0], d2[1], d2[2]]
    # cc = np.linalg.norm(p6 - p7)
    # aa = d3[2]
    # bb = d4[0]
    # alpha = np.arccos((aa**2 + bb**2 - cc**2)/(2*aa*bb))
    # joints[2] = -alpha  + np.pi/2
    # print(p6)
    # print( f"p4: {p4}")
    # # print(f"d2 : {d2}")



    

    
    
    
    joints = np.rad2deg(joints)

    return joints


if __name__ == '__main__':
    # Test
    feetech = Feetech("/dev/ttyACM0")
    fig, ax = create_plot()
    # ax = None
    

    while True:
        joints1 = feetech.get_joints()
        pose = fk(joints1, ax=ax)
        joints2 = ik(pose, ax=ax)

        result = [np.abs(joints1[i]- joints2[i]) <= 0.001 for i in range(5)]
        print(joints1[:5] - joints2)
        print(joints1)
        print(joints2)
        print(result)
        print("______________")
        time.sleep(1)



    plt.show()
