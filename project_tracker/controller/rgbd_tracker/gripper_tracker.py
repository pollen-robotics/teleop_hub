from typing import Optional  # type: ignore

import numpy as np
from controller.filter.filters import KalmanFilter3D, MedianFilter  # type: ignore
from controller.rgbd_tracker.computer_vision import ComputerVision  # type: ignore


class GripperTracker:
    def __init__(self, arm: str, computerVision: ComputerVision) -> None:
        self.computerVision = computerVision
        self.arm = arm

        self.kf = [KalmanFilter3D() for _ in range(4)]

        self.mf_gripper = MedianFilter()

    def get_hand_points(self) -> np.ndarray:
        """Get the 3D coordinates of the hand points (thumb mcp, thumb tip, index mcp, index tip)
        for the specified arm.

        Returns:
            np.ndarray: A numpy array containing the 3D coordinates of the hand points.
        """
        if self.arm == "l_arm":
            return self.computerVision.left_hand_points
        else:
            return self.computerVision.right_hand_points

    def estimate_gripper_opening(self, hand_points) -> Optional[float]:
        """Estimate the gripper opening based on the hand points.

        The gripper opening is calculated as the ratio of the distance between the index tip and thumb tip
        to the distance between the index mcp and thumb mcp.

        Args:
            hand_points (np.ndarray): A numpy array containing the 3D coordinates of the hand points.
        Returns:
            Optional[float]: The estimated gripper opening. Returns None if hand points are not available.
        """
        if not np.any(hand_points):
            return None
        thumb_mcp = hand_points[0]
        thumb_tip = hand_points[1]
        index_mcp = hand_points[2]
        index_tip = hand_points[3]

        opening = float(np.linalg.norm(index_tip - thumb_tip) / np.linalg.norm(index_mcp - thumb_mcp))
        opening_filtered = self.mf_gripper.update(opening)
        return opening_filtered

    def update_gripper_opening(self):
        """Update the gripper opening based on the hand points."""
        hand_points = self.get_hand_points()
        # Apply Kalman filter to smooth the hand points
        hand_points_filtered = [self.kf[i].update(hand_points[i]) for i in range(4)]
        # get the rotation matrix
        gripper_opening = self.estimate_gripper_opening(hand_points_filtered)

        return gripper_opening
