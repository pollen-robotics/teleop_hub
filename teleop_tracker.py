import time
import sys
import numpy as np
import math
import numpy.typing as npt
import triad_openvr
import openvr

from reachy2_sdk import ReachySDK  # type: ignore
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value

from reachy2_sdk.utils.utils import recompose_matrix, get_pose_matrix  # type: ignore
from reachy2_sdk_api.arm_pb2 import (  # type: ignore
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from scipy.spatial.transform import Slerp  # type: ignore


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




class RobotController:
    def __init__(self, host_ip='localhost', arm_side='right', tracker=ViveTracker('tracker_1')):
        self.tracker = tracker
        self.reachy = ReachySDK(host_ip)
        self.reachy.turn_on()
        self.arm_side = arm_side
        self.arm = self.reachy.r_arm if arm_side == 'right' else self.reachy.l_arm

        self.R_vive_to_robot = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ])

    def change_arm_side(self):
        self.arm = self.reachy.l_arm if self.arm == self.reachy.r_arm else self.reachy.r_arm

    def convert_to_robot_frame(self):
        relative_pose = self.tracker.get_relative_tracker_pose()
        rotation_vive = relative_pose[:3, :3]
        rotation_robot = self.R_vive_to_robot @ rotation_vive @ self.R_vive_to_robot.T

        position_vive = relative_pose[:3, 3]
        position_robot = self.R_vive_to_robot @ position_vive

        relative_pose_robot = np.eye(4)
        relative_pose_robot[:3, :3] = rotation_robot
        relative_pose_robot[:3, 3] = position_robot

        print(f"vive frame : {relative_pose[:3,3]}, robot frame :", position_robot)

        return relative_pose_robot


    def go_to_pose(self, pose: npt.NDArray[np.float64]) -> None:
        request = ArmCartesianGoal(
            id=self.reachy.l_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.UNFREEZE,
            constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.05),
            order_id=Int32Value(value=5),
        )
        self.arm._stub.SendArmCartesianGoal(request)


if __name__ == '__main__':
    tracker = ViveTracker('tracker_1')
    # robot = RobotController(tracker=tracker)
    # robot.convert_to_robot_frame()

    while True:
        tracker.update_rpy()
        # goal_pose = robot.convert_to_robot_frame()
        # robot.go_to_pose(goal_pose)