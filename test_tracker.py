import time
import sys
import numpy as np
import math
import numpy.typing as npt
import triad_openvr
import openvr


from utils import make_homogenous_matrix_from_rotation_matrix, create_plot, plot_orientation, update_plot, fk


class ViveTracker:
    def __init__(self, tracker_id):
        self.vive = triad_openvr.triad_openvr()
        self.tracker_name = tracker_id
        self.tracker = self.vive.devices[self.tracker_name]
        self.tracker_euler_angles = np.zeros(3)
        self.tracker_position = np.zeros(3)
        self.tracker_pose = np.eye(4)
        self.zero_pose = np.eye(4)

    def convert_openvr_matrix(self, hmd_matrix):
        """
        Convertit une matrice OpenVR 3x4 (HmdMatrix34_t) en une matrice 4x4 NumPy.

        :param hmd_matrix: Matrice OpenVR (hmd_matrix.mDeviceToAbsoluteTracking)
        :return: Matrice 4x4 NumPy
        """
        m = np.array([
            [hmd_matrix[0][0], hmd_matrix[0][1], hmd_matrix[0][2], hmd_matrix[0][3]],
            [hmd_matrix[1][0], hmd_matrix[1][1], hmd_matrix[1][2], hmd_matrix[1][3]],
            [hmd_matrix[2][0], hmd_matrix[2][1], hmd_matrix[2][2], hmd_matrix[2][3]],
            [0, 0, 0, 1]
        ])
        return m

    def update_tracker_pose(self):
        pose = self.tracker.get_pose_matrix()
        # pose2 = self.tracker.get_pose_euler()
        # print(pose2)

        # pose = np.array(pose)
        # position = pose[:3, 3]
        # orientation = R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True)
        # print(f"position: {position}")
        # print(f"orientation: {orientation}")
        self.tracker_pose = self.convert_openvr_matrix(pose)

    def update_rpy(self):
        [x, y, z, roll, pitch, yaw] = self.tracker.get_pose_euler()
        
        self.tracker_euler_angles = np.array(np.degrees([roll, pitch, yaw]))
        print (self.tracker_euler_angles)


    def get_tracker_position(self) -> npt.ArrayLike:
        return self.tracker_position

    def get_tracker_euler_angles(self) -> npt.ArrayLike:
        return self.tracker_euler_angles

    def calibrate_pose_zero(self):
        print("Calibration : reset zero pose ")
        time.sleep(2)
        self.update_tracker_pose()
        self.zero_pose = self.tracker_pose
        # R_roll = np.array([
        #     [1, 0, 0],
        #     [0, 0, 1],
        #     [0, -1, 0]
        # ])

        # self.center_pose[:3, :3] = self.tracker_pose[:3, :3]

        # print("Calibration : Move the tracker in front of your sternum")
        # user_positions = []
        # time.sleep(2)
        # t0 = time.time()
        # while time.time() - t0 < 3 or len(user_positions) < 10:
        #     user_positions.append(self.tracker.get_pose_euler()[:3])
        #     time.sleep(0.1)
        # mean_position = np.median(user_positions, axis=0)
        
        # self.center_pose[:3,3] = mean_position
        # print("center pose", self.center_pose)


    def get_relative_tracker_pose(self):
        self.update_tracker_pose()
        relative_pose = np.linalg.inv(self.center_pose) @ self.tracker_pose
        relative_position = relative_pose[:3, 3]
        # print("relative_position", relative_position)
        # print("relative orientation in euler angles", R.from_matrix(relative_pose[:3, :3]).as_euler('xyz', degrees=True))
        return relative_pose



if __name__ == '__main__':
    tracker = ViveTracker('tracker_1')
    # robot = RobotController(tracker=tracker)
    # robot.convert_to_robot_frame()

    while True:
        tracker.update_rpy()
        pose = tracker.tracker_pose
        print(pose)
        # goal_pose = robot.convert_to_robot_frame()
        # robot.go_to_pose(goal_pose)