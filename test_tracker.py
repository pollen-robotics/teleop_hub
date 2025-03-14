import math
import sys
import time

import numpy as np
import numpy.typing as npt
import openvr
from scipy.spatial.transform import Rotation as R

import triad_openvr
from utils import (
    create_plot,
    fk,
    make_homogenous_matrix_from_rotation_matrix,
    plot_orientation,
    update_plot,
)


class ViveTracker:
    def __init__(self, tracker_id):
        self.vive = triad_openvr.triad_openvr()
        self.tracker_name = tracker_id
        self.tracker = self.vive.devices[self.tracker_name]
        self.tracker_euler_angles = np.zeros(3)
        self.tracker_position = np.zeros(3)
        self.tracker_pose = np.eye(4)  # Current pose
        self.zero_pose = None  # Initial reference frame

    def convert_openvr_matrix(self, hmd_matrix):
        """
        Convert an OpenVR 3x4 matrix (HmdMatrix34_t) to a 4x4 NumPy matrix.

        :param hmd_matrix: OpenVR matrix (hmd_matrix.mDeviceToAbsoluteTracking)
        :return: 4x4 NumPy transformation matrix
        """
        m = np.array(
            [
                [
                    hmd_matrix[0][0],
                    hmd_matrix[0][1],
                    hmd_matrix[0][2],
                    hmd_matrix[0][3],
                ],
                [
                    hmd_matrix[1][0],
                    hmd_matrix[1][1],
                    hmd_matrix[1][2],
                    hmd_matrix[1][3],
                ],
                [
                    hmd_matrix[2][0],
                    hmd_matrix[2][1],
                    hmd_matrix[2][2],
                    hmd_matrix[2][3],
                ],
                [0, 0, 0, 1],
            ]
        )
        return m

    def update_tracker_pose(self):
        """Updates the tracker pose in the SteamVR world frame."""
        pose = self.tracker.get_pose_matrix()
        self.tracker_pose = self.convert_openvr_matrix(pose)

    def calibrate_pose_zero(self):
        """Sets the current pose as the zero reference frame."""
        print("Calibration: Setting the initial pose as the new coordinate system...")
        time.sleep(2)
        self.update_tracker_pose()
        self.zero_pose = self.tracker_pose  # Store initial pose as reference

    def get_relative_tracker_pose(self):
        """Computes the tracker pose relative to the initial calibration pose."""
        self.update_tracker_pose()

        if self.zero_pose is None:
            print("Warning: Zero pose not set. Returning absolute pose.")
            return self.tracker_pose  # Return absolute pose if not calibrated

        relative_pose = np.linalg.inv(self.zero_pose) @ self.tracker_pose

        rotation = R.from_euler("xyz", [0, 0, -30], degrees=True).as_matrix()
        Trot = make_homogenous_matrix_from_rotation_matrix(
            rotation, [0, 0, 0])
        relative_pose = Trot @ relative_pose

        rotation = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
        Trot = make_homogenous_matrix_from_rotation_matrix(
            rotation, [0, 0, 0])
        relative_pose = Trot @ relative_pose

        return relative_pose

    def get_tracker_position(self) -> npt.ArrayLike:
        """Returns the tracker's current position relative to the calibrated frame."""
        relative_pose = self.get_relative_tracker_pose()
        return relative_pose[:3, 3]

    def get_tracker_euler_angles(self) -> npt.ArrayLike:
        """Returns the tracker's orientation (RPY) relative to the calibrated frame."""
        relative_pose = self.get_relative_tracker_pose()
        return R.from_matrix(relative_pose[:3, :3]).as_euler("xyz", degrees=True)


if __name__ == "__main__":
    tracker = ViveTracker("tracker_1")

    # Set new coordinate system
    tracker.calibrate_pose_zero()

    while True:
        position = tracker.get_tracker_position()
        orientation = tracker.get_tracker_euler_angles()

        print(f"Relative position: {position}")
        # print(f"Relative orientation (Euler XYZ): {orientation}")

        time.sleep(0.1)
