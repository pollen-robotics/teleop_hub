import time
from abc import ABC
from collections import deque
from typing import Deque, Optional

import numpy as np
from controller.controller import Controller  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from trackers.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from trackers.rgbd_tracker.rgbd_tracker import ArmRGBDTracker  # type: ignore
from trackers.rgbd_tracker.rgbd_tracker import GripperRGBDTracker, HeadRGBDTracker
from utils import make_homogenous_matrix_from_rotation_matrix  # type: ignore


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
        """Initialize user parameters for the RGBD controller
        (e.g., user center and normalization factor from pixels to meters).
        """
        while not np.any(self.computer_vision.user_center):
            time.sleep(0.2)
        self.user_center = self.computer_vision.user_center
        self.normalization_factor = 1 / self.computer_vision.normalization_factor

    def get_controller_pose(self):
        """Get the pose of the RGBD tracker.

        This method updates the tracker pose and converts it from the image to the robot frame.

        Returns:
            np.ndarray: The pose of the RGBD tracker.
        """
        pass


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

        # Parameters for stopping the teleoperation
        self.former_rpy: Deque = deque(maxlen=10)
        self.stop_flag = False

    def get_controller_pose(self) -> np.ndarray:
        """Get the pose of the RGBD Head tracker.

        This method updates the head pose and converts it from the image to the robot frame.
        It also checks if the stop flag is raised.

        Returns:
            np.ndarray: The pose of the head in the robot frame.
        """
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose
        pose = self.convert_pose(pose)
        self.check_stop_flag(pose)

        return pose

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
        return rotation

    def check_stop_flag(self, rotation: np.ndarray) -> None:
        """Check if the user wants to stop the teleoperation.

        This method adds the RPY angles to a deque, and stop the tracking if the stop flag is raised.

        Args:
            rotation (np.ndarray): The rotation matrix of the head RGBD tracker.
        """
        rpy = R.from_matrix(rotation).as_euler("XYZ", degrees=True)

        # Add the current rpy to the deque, to check if the user wants to stop the teleoperation
        self.former_rpy.append(rpy)
        if self.raise_flag_to_stop():
            print("Command to stop")
            self.stop_flag = True
            self.tracker.stop()

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
        """Initialize the arm RGBD controller, with an ArmRGBDTracker and a GripperRGBDTracker.

        Args:
            computer_vision (ComputerVision): The computer vision object used for tracking.
            arm (str): The arm to be controlled (e.g., "l_arm" or "r_arm").
        """
        super().__init__(computer_vision)
        self.arm = arm
        self.tracker = ArmRGBDTracker(self.arm, self.computer_vision)
        self.gripper = GripperRGBDTracker(self.arm, self.computer_vision)

        self.first_pose: Optional[np.ndarray] = None
        self.former_pose: Optional[np.ndarray] = None

    def get_controller_pose(self) -> np.ndarray:
        """Get the pose of the RGBD Arm tracker.

        This method updates the arm pose and converts it from the image to the robot frame.
        It also checks if the command is reachable and if it's not too far from the previous one,
        otherwise the command is ignored.

        Returns:
            np.ndarray: The pose of the arm in the robot frame.
        """
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose
        pose = self.convert_pose(pose)

        # Check if the pose is reachable
        if not self.is_command_reachable(pose):
            pose = self.former_pose
            print("Pose unreachable, using former pose.")

        # Check if the pose is too far from the previous one
        distance_threshold = 0.2
        if self.former_pose is not None:
            if self.is_too_far(pose, distance_threshold):
                pose = self.former_pose
                print("Pose too far from the previous one, using former pose.")

        self.former_pose = pose
        return pose

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

        new_pose = make_homogenous_matrix_from_rotation_matrix(
            new_rotation, new_position
        )

        return new_pose

    def check_first_command(self, pose: np.ndarray) -> bool:
        """Check if the first command is in a specific area, and save it if it is.

        That method is to avoid to have an aberrant command
        that could cause the robot to move abruptly.

        Args:
            pose (np.ndarray): The pose of the controller.
        Returns:
            bool: True if the first command is valid, False otherwise.
        """
        if pose is None:
            return False

        command = pose.copy()[:3, 3]
        if self.arm == "r_arm":
            command[1] = -command[1]

        # check if the first command is in the cube : x [0.2, O.4], y [0.2, 0.4], z [-0.4,-0.1]
        if (
            command[0] < 0.6
            and command[0] > 0.1
            and command[1] < 0.5
            and command[1] > 0.1
            and command[2] < 0
            and command[2] > -0.5
        ):
            self.first_pose = pose
            self.former_pose = pose
            return True

        print(f"First command of {self.arm} : {command} is not in the cube")
        return False

    def is_too_far(self, goal_pose: np.ndarray, distance_threshold: float) -> bool:
        """Check if the goal pose is too far from the former pose.

        This method is used to avoid abrupt movements of the robot.

        Args:
            goal_pose (np.ndarray): The goal pose of the controller.
            distance_threshold (float): The distance threshold to check if the goal pose is too far.
        Returns:
            bool: True if the goal pose is too far from the former pose, False otherwise.
        """
        goal_position = goal_pose[:3, 3]
        former_position = self.former_pose[:3, 3]
        return bool(
            np.linalg.norm(goal_position - former_position) > distance_threshold
        )

    def is_command_reachable(self, goal_pose: np.ndarray) -> bool:
        """Check if the goal pose is reachable.

        This method is used to avoid commands that are too far from the robot.

        Args:
            goal_pose (np.ndarray): The goal pose of the controller.
        Returns:
            bool: True if the goal pose is reachable, False otherwise.
        """
        goal_position = goal_pose[:3, 3]
        if (
            goal_position[0] < 0.1
            or goal_position[0] > 0.8
            or goal_position[1] < -0.6
            or goal_position[1] > 0.6
            or goal_position[2] < -0.6
            or goal_position[2] > 0.4
        ):
            print(goal_position)
            return False
        return True

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
