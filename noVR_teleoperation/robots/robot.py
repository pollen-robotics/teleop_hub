from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Robot(ABC):
    """Abstract base class for a robot.

    This class defines the interface for initializing, controlling, and stopping a robot.
    """

    def __init__(self, robot_ip: str, mirror_mode: bool = False) -> None:
        """Initializes the robot.

        Args:
            robot_ip (str): The IP address of the robot.
            mirror_mode (bool): If True, the robot will operate in mirror mode
                (e.g., left controller controls the right arm). Defaults to False.
        """
        self.robot_ip: str = robot_ip
        self.mirror_mode = mirror_mode

    @abstractmethod
    def init_robot(self):
        pass

    @abstractmethod
    def fk(self, arm: str) -> np.ndarray:
        """Compute the forward kinematics of the specified arm.

        Args:
            arm (str): The specified arm. Can be "r_arm" or "l_arm".
        Returns:
            np.ndarray: The effector pose of the specified arm as a 4x4 transformation matrix.
        """
        pass

    @abstractmethod
    def go_to_pose(self, pose: np.ndarray, arm: str) -> None:
        """Send the command to move the specified arm to the target pose.

        Args:
            pose (np.ndarray): The target pose as a 4x4 transformation matrix.
            arm (str): The specified arm. Can be "r_arm" or "l_arm".
        """
        pass

    @abstractmethod
    def move_gripper(
        self, arm: str, with_joint_command: bool = True, command: Optional[float] = None
    ) -> None:
        """Send the command to move the gripper of the specified arm.

        Args:
            arm (str): The specified arm. Can be "r_arm" or "l_arm".
            with_joint_command (bool): If True, the command is sent as a joint command.
                If False, the command is sent as an open/close command.
            command (Optional[float]): The command to send to the gripper.
                If with_joint_command is True, this should be a float value between 0 and 100.
                If with_joint_command is False, this should be 0 (close) or 1 (open).
        """
        pass

    @abstractmethod
    def move_mobile_base(self, x: float, y: float, theta: float) -> None:
        """Send the command to move the mobile base.

        Args:
            x (float): The target x position.
            y (float): The target y position.
            theta (float): The target orientation in degrees.
        """
        pass

    @abstractmethod
    def move_antenna(self, position: float, arm=None) -> None:
        """Send the command to move the antenna, according to the controller mode.

        Args:
            - position (float): The target position in degrees.
            - arm (Optional[str]): The side of the controller that controls the antenna.
                Can be "r_arm" or "l_arm". If None, both antennas are moved.
        """
        pass

    @abstractmethod
    def move_head(self, orientation_matrix: np.ndarray) -> None:
        """Send the command to move the head.

        Args:
            - orientation_matrix (np.ndarray): The target orientation as a 3x3 rotation matrix.
        """
        pass

    @abstractmethod
    def stop(self):
        pass
