import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2  # type: ignore
import numpy as np
from camera.orbbec import Orbbec  # type: ignore
from filter.filters import (  # type: ignore
    KalmanFilter3D,
    MedianFilter,
    RotationSmoother,
)
from reachy2_sdk.utils.utils import recompose_matrix  # type: ignore
from trackers.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from trackers.tracker import Tracker, TrackerType  # type: ignore
from utils import rotation_matrix_from_vector  # type: ignore


class RGBDTracker(Tracker, ABC):
    """Abstract class for RGBD trackers.

    This class serves as a base for specific RGBD trackers, such as arm, gripper, and head trackers.
    It inherits from the Tracker class and implements common functionality for RGBD tracking.
    """

    def __init__(self, computer_vision: ComputerVision) -> None:
        """Initialize the RGBD tracker.

        Args:
            computer_vision (ComputerVision): An instance of the ComputerVision class for processing RGBD data.
        """
        super().__init__()
        self.computer_vision = computer_vision
        self.tracker_type = TrackerType.RGBD
        self.tracker_pose: Optional[np.ndarray]

    @abstractmethod
    def get_points(self):
        """Get the 3D coordinates of the tracked points.

        This method should be implemented by subclasses to return the specific points of interest
        for the corresponding tracker type.
        """
        pass


class ArmRGBDTracker(RGBDTracker):
    """RGBD tracker for arm tracking.

    This class inherits from the RGBDTracker class and implements functionality for tracking arm positions
    using RGBD data.
    """

    def __init__(self, arm: str, computer_vision: ComputerVision) -> None:
        """Initialize the ArmRGBDTracker.

        Args:
            arm (str): The arm to be tracked ("l_arm" or "r_arm").
            computer_vision (ComputerVision): An instance of the ComputerVision class for processing RGBD data.
        """
        super().__init__(computer_vision)
        self.arm = arm
        self.kalman_filters = [KalmanFilter3D() for _ in range(2)]
        self.rotation_smoother = RotationSmoother()

    def update_tracker_pose(self) -> None:
        """Update the tracker pose based on the arm points.

        This method uses the wrist position as the goal position,
        retrieves the elbow and wrist positions, applies Kalman filtering to smooth the positions,
        and computes the rotation matrix from the vector elbow-wrist.
        """
        elbow_position, wrist_position = self.get_points()

        #  Apply Kalman filter to smooth the elbow and wrist positions
        elbow_position_filtered = self.kalman_filters[0].update(elbow_position)
        wrist_position_filtered = self.kalman_filters[1].update(wrist_position)

        # get the rotation matrix from the vector elbow-wrist
        wrist_rotation = rotation_matrix_from_vector(wrist_position_filtered - elbow_position_filtered)
        wrist_rotation_filtered = self.rotation_smoother.update(wrist_rotation)

        self.tracker_pose = recompose_matrix(wrist_rotation_filtered, wrist_position_filtered)

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

    def stop(self) -> None:
        """Stop the RGBD tracker."""
        self.computer_vision.stop()
        print("RGBD Tracker stopped.")


class GripperRGBDTracker(RGBDTracker):
    """RGBD tracker for gripper tracking.

    This class inherits from the RGBDTracker class and implements functionality for tracking gripper positions
    using RGBD data.
    """

    def __init__(self, arm: str, computer_vision: ComputerVision) -> None:
        """Initialize the GripperRGBDTracker.

        Args:
            arm (str): The arm to be tracked ("l_arm" or "r_arm").
            computer_vision (ComputerVision): An instance of the ComputerVision class for processing RGBD data.
        """
        super().__init__(computer_vision)
        self.arm = arm
        self.kf = [KalmanFilter3D() for _ in range(4)]
        self.mf_gripper = MedianFilter()

    def update_tracker_pose(self) -> Optional[float]:
        """Update the gripper opening based on the hand points.

        This method retrieves the hand points, applies Kalman filtering to smooth the positions,
        and computes the gripper opening.

        Returns:
            Optional[float]: The estimated gripper opening. Returns None if hand points are not available.
        """
        hand_points = self.get_points()
        # Apply Kalman filter to smooth the hand points
        hand_points_filtered = [self.kf[i].update(hand_points[i]) for i in range(4)]
        # get the rotation matrix
        gripper_opening = self._estimate_gripper_opening(hand_points_filtered)

        return gripper_opening

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

    def _estimate_gripper_opening(self, hand_points) -> Optional[float]:
        """Estimate the gripper opening based on the hand points.

        The gripper opening is calculated as the ratio of the distance between the index tip and thumb tip
        to the distance between the index mcp and thumb mcp.
        A median filter is applied to smooth the opening value.

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


class HeadRGBDTracker(RGBDTracker):
    """RGBD tracker for head tracking.

    This class inherits from the RGBDTracker class and implements functionality for tracking head positions
    using RGBD data.
    """

    def __init__(self, computer_vision: ComputerVision) -> None:
        """Initialize the HeadRGBDTracker.
        Args:
            computer_vision (ComputerVision): An instance of the ComputerVision class for processing RGBD data.
        """
        super().__init__(computer_vision)
        self.kf = [KalmanFilter3D() for _ in range(4)]
        self.rotation_smoother = RotationSmoother()

    def update_tracker_pose(self) -> None:
        """Update the head pose based on the head points.

        This method retrieves the head points, applies Kalman filtering to smooth the positions,
        and computes the head orientation.
        """
        head_points = self.get_points()
        # Apply Kalman filter to smooth the head points
        head_points_filtered = [self.kf[i].update(head_points[i]) for i in range(4)]
        # get the rotation matrix
        tracker_pose = self._estimate_head_orientation(head_points_filtered)
        self.tracker_pose = tracker_pose

    def get_points(self) -> np.ndarray:
        """Get the 3D coordinates of the face points (nose, chin, left eye, right eye).

        Returns:
            np.ndarray: A numpy array containing the 3D coordinates of the face points.
        """
        face_points = self.computer_vision.face_points
        return face_points

    def _estimate_head_orientation(self, head_points: list) -> np.ndarray:
        """Estimate the head orientation based on the head points.

        The head orientation is calculated using the nose, chin, left eye, and right eye points.

        Args:
            head_points (np.ndarray): A numpy array containing the 3D coordinates of the head points.
        Returns:
            np.ndarray: The estimated head orientation as a rotation matrix.
        """
        nose = head_points[0]
        chin = head_points[1]
        left_eye = head_points[2]
        right_eye = head_points[3]

        # Create the 3D axes from the points
        x_axis = np.array(right_eye) - np.array(left_eye)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.array(chin) - np.array(nose)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        rot = np.array([x_axis, y_axis, z_axis]).T

        return rot


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
