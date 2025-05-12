import time
from abc import ABC
from collections import deque
from typing import Deque, Optional

import numpy as np
from controller.controller import Controller  # type: ignore
from reachy2_sdk.utils.utils import recompose_matrix  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from trackers.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from trackers.rgbd_tracker.rgbd_tracker import (  # type: ignore
    ArmRGBDTracker,
    GripperRGBDTracker,
    HeadRGBDTracker,
)


class RGBDController(Controller, ABC):
    """Abstract base class for all RGBD controllers.

    This class defines the interface for a controller that interacts with a RGBD tracker.
    It is a subclass of the Controller class.
    """

    def __init__(self, computer_vision: ComputerVision) -> None:
        """Initialize the RGBD controller.
        Args:
            computer_vision (ComputerVision): The computer vision object used for tracking.
        """
        super().__init__()
        self.computer_vision = computer_vision
        self.init_user_parameters()

    def init_user_parameters(self) -> None:
        """Initialize user parameters for the RGBD controller."""
        while not np.any(self.computer_vision.user_center):
            time.sleep(0.2)
        self.user_center = self.computer_vision.user_center
        real_dist_intershoulder = 0.3
        self.normalization_factor = (
            real_dist_intershoulder / self.computer_vision.dist_intershoulder
        )

    def get_controller_pose(self) -> np.ndarray:
        """Get the pose of the RGBD tracker.
        This method updates the tracker pose and converts it from the image to the robot frame.

        Returns:
            np.ndarray: The pose of the RGBD tracker.
        """
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose
        pose = self.convert_pose(pose)
        return pose


class HeadRGBDController(RGBDController):
    """Head RGBD controller class.

    This class is responsible for controlling the head RGBD tracker.
    It is a subclass of the RGBDController class.
    """

    def __init__(self, computer_vision: ComputerVision) -> None:
        """Initialize the head RGBD controller.

        Args:
            computer_vision (ComputerVision): The computer vision object used for tracking.
        """
        super().__init__(computer_vision)
        self.tracker = HeadRGBDTracker(self.computer_vision)
        self.former_rpy: Deque = deque(maxlen=10)
        self.stop_flag = False

    def convert_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert the pose from the image to the robot frame.
        This method corrects the pose using the camera transformation matrix and
        applies a transformation to the pose.

        Args:
            pose (np.ndarray): The pose to be converted.
        Returns:
            np.ndarray: The converted pose.
        """
        corrected_pose = self.computer_vision.T_world_camera @ pose
        T_cam_to_reachy = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        rotation = T_cam_to_reachy @ corrected_pose @ T_cam_to_reachy.T
        rpy = R.from_matrix(rotation).as_euler("XYZ", degrees=True)

        # Add the current rpy to the deque, to check if the user wants to stop the teleoperation
        self.former_rpy.append(rpy)
        if self.raise_flag_to_stop():
            print("Command to stop")
            self.stop_flag = True

        return rotation

    def raise_flag_to_stop(self) -> bool:
        """Check if the user wants to stop the teleoperation.

        This method checks the last 10 poses of the head RGBD tracker : if the user is looking down
        for a prolonged period, it raises a flag to stop the teleoperation.

        Returns:
            bool: True if the flag to stop the teleoperation is raised, False otherwise.
        """
        if len(self.former_rpy) == 10:
            pitch_values = [rpy[1] for rpy in self.former_rpy]
            roll_values = [rpy[0] for rpy in self.former_rpy]
            yaw_values = [rpy[2] for rpy in self.former_rpy]
            if (
                np.all(np.abs(pitch_values) > 25)
                and np.all(np.abs(roll_values) < 10)
                and np.all(np.abs(yaw_values) < 10)
            ):
                print("Command to stop")
                return True
        return False


class ArmRGBDController(RGBDController):
    """Arm RGBD controller class.

    This class is responsible for controlling the arm RGBD tracker.
    It is a subclass of the RGBDController class.
    """

    def __init__(self, computer_vision: ComputerVision, arm: str) -> None:
        """Initialize the arm RGBD controller.
        Args:
            computer_vision (ComputerVision): The computer vision object used for tracking.
            arm (str): The arm to be controlled (e.g., "l_arm" or "r_arm").
        """
        super().__init__(computer_vision)
        self.arm = arm
        self.tracker = ArmRGBDTracker(self.arm, self.computer_vision)
        self.gripper = GripperRGBDTracker(self.arm, self.computer_vision)
        self.stop_flag = False

    def convert_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert the pose from the image to the robot frame.

        This method corrects the pose using the camera transformation matrix and
        applies a transformation to the pose.

        Args:
            pose (np.ndarray): The pose to be converted.
        Returns:
            np.ndarray: The converted pose.
        """
        corrected_rotation = self.computer_vision.T_world_camera @ pose[:3, :3]
        corrected_position = self.computer_vision.T_world_camera @ (
            pose[:3, 3] - self.user_center
        )

        normalized_position = corrected_position * self.normalization_factor

        x = normalized_position[0]
        y = normalized_position[1]
        z = normalized_position[2]

        new_position = np.array([-z, x, -y])

        T_cam_to_reachy = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        new_rotation = T_cam_to_reachy @ corrected_rotation @ T_cam_to_reachy.T

        new_pose = recompose_matrix(new_rotation, new_position)

        return new_pose

    def get_gripper_command(self) -> Optional[float]:
        """Get the gripper command.

        This method updates the distance between index and thumb,
        and determines the gripper command based on the opening value.

        Returns:
            float: The gripper command (0 for close, 1 for open, None for no command).
        """
        opening = self.gripper.update_tracker_pose()
        if opening is None:
            return None

        if opening < 1:
            command = 0
        elif opening > 1.2:
            command = 1
        else:
            command = None

        return command

    def stop(self):
        self.stop_flag = True
        self.tracker.stop()
