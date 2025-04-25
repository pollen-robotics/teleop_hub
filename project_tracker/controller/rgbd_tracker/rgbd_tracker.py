import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2  # type: ignore
import numpy as np
from camera.orbbec import Orbbec  # type: ignore
from controller.filter.filters import (  # type: ignore
    KalmanFilter3D,
    MedianFilter,
    RotationSmoother,
)
from controller.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from controller.tracker import Tracker, TrackerType  # type: ignore
from reachy2_sdk.utils.utils import recompose_matrix  # type: ignore
from utils import rotation_matrix_from_vector  # type: ignore


class RGBDTracker(Tracker, ABC):
    def __init__(self, arm: str, computer_vision: ComputerVision) -> None:
        super().__init__(arm)
        self.computer_vision = computer_vision
        self.tracker_type = TrackerType.RGBD
        self.tracker_pose = None

    @abstractmethod
    def get_points(self) -> np.ndarray:
        pass


class ArmRGBDTracker(RGBDTracker):
    def __init__(self, arm: str, computer_vision: ComputerVision) -> None:
        super().__init__(arm, computer_vision)
        self.kalman_filters = [KalmanFilter3D() for _ in range(2)]
        self.rotation_smoother = RotationSmoother()

    def get_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the 3D coordinates of the arm points (elbow, wrist) for the specified arm.

        Returns:
            tuple: A tuple containing the 3D coordinates of the elbow and wrist points.
        """
        side_int = 0 if self.arm == "l_arm" else 1
        return (
            np.array(self.computer_vision.elbows[side_int]),
            np.array(self.computer_vision.wrists[side_int]),
        )

    def update_tracker_pose(self):
        elbow_position, wrist_position = self.get_points()
        # Apply Kalman filter to smooth the elbow and wrist positions
        elbow_position_filtered = self.kalman_filters[0].update(elbow_position)
        wrist_position_filtered = self.kalman_filters[1].update(wrist_position)

        # get the rotation matrix from the vector elbow-wrist
        wrist_rotation = rotation_matrix_from_vector(wrist_position_filtered - elbow_position_filtered)
        wrist_rotation_filtered = self.rotation_smoother.update(wrist_rotation)

        self.tracker_pose = recompose_matrix(wrist_rotation_filtered, wrist_position_filtered)

    def stop(self):
        self.computer_vision.stop()
        print("RGBD Tracker stopped.")


class GripperRGBDTracker(RGBDTracker):
    def __init__(self, arm: str, computer_vision: ComputerVision) -> None:
        super().__init__(arm, computer_vision)
        self.kf = [KalmanFilter3D() for _ in range(4)]
        self.mf_gripper = MedianFilter()

    def get_points(self) -> np.ndarray:
        """Get the 3D coordinates of the hand points (thumb mcp, thumb tip, index mcp, index tip)
        for the specified arm.

        Returns:
            np.ndarray: A numpy array containing the 3D coordinates of the hand points.
        """
        if self.arm == "l_arm":
            return self.computer_vision.left_hand_points
        else:
            return self.computer_vision.right_hand_points

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

    def update_tracker_pose(self):
        """Update the gripper opening based on the hand points."""
        hand_points = self.get_points()
        # Apply Kalman filter to smooth the hand points
        hand_points_filtered = [self.kf[i].update(hand_points[i]) for i in range(4)]
        # get the rotation matrix
        gripper_opening = self.estimate_gripper_opening(hand_points_filtered)

        return gripper_opening


class HeadRGBDTracker(RGBDTracker):
    def __init__(self, computer_vision: ComputerVision) -> None:
        super().__init__("head", computer_vision)
        self.kf = [KalmanFilter3D() for _ in range(4)]
        self.rotation_smoother = RotationSmoother()

    def get_points(self):
        face_points = self.computer_vision.face_points
        return face_points

    def estimate_head_orientation(self, head_points) -> np.ndarray:
        nose = head_points[0]
        chin = head_points[1]
        left_eye = head_points[2]
        right_eye = head_points[3]

        # in IMAGE FRAME
        x_axis = np.array(right_eye) - np.array(left_eye)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.array(chin) - np.array(nose)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        rot = np.array([x_axis, y_axis, z_axis]).T

        return rot

    def update_tracker_pose(self):
        head_points = self.get_points()
        # Apply Kalman filter to smooth the head points
        head_points_filtered = [self.kf[i].update(head_points[i]) for i in range(4)]
        # get the rotation matrix
        tracker_pose = self.estimate_head_orientation(head_points_filtered)
        self.tracker_pose = tracker_pose


if __name__ == "__main__":
    camera = Orbbec()
    computer_vision = ComputerVision(camera)
    l_rgbd_tracker = ArmRGBDTracker("l_arm", computer_vision)
    r_rgbd_tracker = ArmRGBDTracker("r_arm", computer_vision)
    time.sleep(3)
    while True:
        success = computer_vision.update_landmarks_coordinates()
        if not success:
            print("No landmarks available.")
        else:
            for tracker in [l_rgbd_tracker, r_rgbd_tracker]:
                tracker.update_tracker_pose()
                if tracker.tracker_pose is not None:
                    elbow, wrist = tracker.get_points()
                    print(f"{tracker.arm}\n Elbow: {elbow}, Wrist: {wrist}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for tracker in [l_rgbd_tracker, r_rgbd_tracker]:
        tracker.stop()

# class RGBDTracker(Tracker):
#     def __init__(self, arm: str, computer_vision: ComputerVision) -> None:
#         super().__init__(arm)
#         self.computer_vision = computer_vision
#         self.tracker_type = TrackerType.RGBD
#         self.tracker_pose = None

#         self.kalman_filters = [KalmanFilter3D() for _ in range(2)]  # elbow and wrist
#         self.rotation_smoother = RotationSmoother()

#     def get_arm_points(self) -> tuple[np.ndarray, np.ndarray]:
#         """Get the 3D coordinates of the arm points (elbow, wrist) for the specified arm.

#         Returns:
#             tuple: A tuple containing the 3D coordinates of the elbow and wrist points.
#         """
#         side_int = 0 if self.arm == "l_arm" else 1
#         return (
#             np.array(self.computer_vision.elbows[side_int]),
#             np.array(self.computer_vision.wrists[side_int]),
#         )

#     def update_tracker_pose(self):
#         elbow_position, wrist_position = self.get_arm_points()
#         # Apply Kalman filter to smooth the elbow and wrist positions
#         elbow_position_filtered = self.kalman_filters[0].update(elbow_position)
#         wrist_position_filtered = self.kalman_filters[1].update(wrist_position)

#         # get the rotation matrix from the vector elbow-wrist
#         wrist_rotation = rotation_matrix_from_vector(wrist_position_filtered - elbow_position_filtered)
#         wrist_rotation_filtered = self.rotation_smoother.update(wrist_rotation)

#         self.tracker_pose = recompose_matrix(wrist_rotation_filtered, wrist_position_filtered)

#     def stop(self):
#         self.computer_vision.stop()
#         print("RGBD Tracker stopped.")


# if __name__ == "__main__":
#     camera = Orbbec()
#     computer_vision = ComputerVision(camera)
#     l_rgbd_tracker = RGBDTracker("l_arm", computer_vision)
#     r_rgbd_tracker = RGBDTracker("r_arm", computer_vision)
#     time.sleep(3)
#     while True:
#         success = computer_vision.update_landmarks_coordinates()
#         if not success:
#             print("No landmarks available.")
#         else:
#             for tracker in [l_rgbd_tracker, r_rgbd_tracker]:
#                 tracker.update_tracker_pose()
#                 if tracker.tracker_pose is not None:
#                     elbow, wrist = tracker.get_arm_points()
#                     print(f"{tracker.arm}\n Elbow: {elbow}, Wrist: {wrist}")
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     for tracker in [l_rgbd_tracker, r_rgbd_tracker]:
#         tracker.stop()
