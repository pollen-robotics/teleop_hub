from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore

from robots.robot import Robot  # type: ignore
from utils import (limit_orbita3d_joints, load_config,  # type: ignore
                   make_homogenous_matrix_from_rotation_matrix)


class Reachy2(Robot):
    """Class for controlling Reachy2.

    This class inherits from the Robot class.
    """

    def __init__(self, robot_ip: str = "localhost", mirror_mode: bool = False) -> None:
        """Instanciates a Reachy2, connecting to the specified IP address.

        Args:
            robot_ip (str): The IP address of the Reachy2 robot. Defaults to "localhost".
            mirror_mode (bool): If True, the robot will operate in mirror mode
            (e.g., left controller controls the right arm). Defaults to False.
        """
        super().__init__(robot_ip, mirror_mode)
        self._make_imports()
        fake_only_parameter = load_config("config.yaml").get("fake_only", True)
        self.reachy = self.ReachySDK(self.robot_ip, fake_only=fake_only_parameter)

    def _make_imports(self) -> None:
        from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
        from reachy2_sdk import ReachySDK  # type: ignore
        from reachy2_sdk_api.arm_pb2 import ArmCartesianGoal  # type: ignore
        from reachy2_sdk_api.arm_pb2 import (  # type: ignore
            IKConstrainedMode,
            IKContinuousMode,
        )
        from reachy2_sdk_api.kinematics_pb2 import Matrix4x4  # type: ignore

        self.ReachySDK = ReachySDK
        self.ArmCartesianGoal = ArmCartesianGoal
        self.IKConstrainedMode = IKConstrainedMode
        self.IKContinuousMode = IKContinuousMode
        self.Matrix4x4 = Matrix4x4
        self.FloatValue = FloatValue
        self.Int32Value = Int32Value

    def init_robot(self) -> None:
        """Initializes the robot.

        This method turns on its components, resets the odometry and sets its initial pose.
        """
        self.reachy.turn_on()

        self.reachy.reset_default_limits()
        if self.reachy.mobile_base is not None:
            self.reachy.mobile_base.reset_odometry()

        self._goto_initial_pose()

    def _goto_initial_pose(self) -> None:
        """Moves the robot to its initial pose."""
        # Open grippers
        self.reachy.r_arm.gripper.open()
        self.reachy.l_arm.gripper.open()

        # Set initial pose for arms
        arm_orientation = R.from_euler("xyz", [0, -np.pi / 2, 0], degrees=False)
        arm_position = [0.36, 0.2, -0.28]

        for arm in [self.reachy.l_arm, self.reachy.r_arm]:
            arm_position[1] = (
                -arm_position[1] if arm == self.reachy.r_arm else arm_position[1]
            )
            pose = make_homogenous_matrix_from_rotation_matrix(
                arm_orientation.as_matrix(), arm_position
            )
            joints = arm.inverse_kinematics(pose)
            arm.goto(
                joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False
            )

        # Set initial pose for head and antennas
        self.reachy.head.r_antenna.goto(
            0, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False
        )
        self.reachy.head.l_antenna.goto(
            0, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False
        )
        head_joints = [0, 0, 0]
        self.reachy.head.goto(
            head_joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True
        )

    def fk(self, arm) -> np.ndarray:
        """Compute the forward kinematics of the specified arm.

        Args:
            arm (str): The specified arm. Can be "r_arm" or "l_arm".
        Returns:
            np.ndarray: The effector pose of the specified arm as a 4x4 transformation matrix.
        """
        if arm == "r_arm":
            return self.reachy.r_arm.forward_kinematics()
        else:
            return self.reachy.l_arm.forward_kinematics()

    def unfreeze(self, arm: str) -> None:
        """Send an unfreeze command to the specified arm. Only for the first move.

        Args:
            arm (str): The specified arm. Can be "r_arm" or "l_arm".

        """
        pose = self.fk(arm)
        robot_arm = self.reachy.r_arm if arm == "r_arm" else self.reachy.l_arm

        request = self.ArmCartesianGoal(
            id=robot_arm._part_id,
            goal_pose=self.Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=self.IKContinuousMode.UNFREEZE,
            constrained_mode=self.IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=self.FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=self.FloatValue(value=0.05),
            order_id=self.Int32Value(value=5),
        )
        robot_arm._stub.SendArmCartesianGoal(request)

    def go_to_pose(self, pose: np.ndarray, arm: str) -> None:
        """Send the command to move the specified arm to the target pose.

        Args:
            pose (np.ndarray): The target pose as a 4x4 transformation matrix.
            arm (str): The specified arm. Can be "r_arm" or "l_arm".
        """
        if not self.mirror_mode:
            if arm == "r_arm":
                robot_arm = self.reachy.r_arm
            else:
                robot_arm = self.reachy.l_arm

        else:
            pose = self.get_mirror_pose(pose)
            if arm == "r_arm":
                robot_arm = self.reachy.l_arm
            else:
                robot_arm = self.reachy.r_arm

        request = self.ArmCartesianGoal(
            id=robot_arm._part_id,
            goal_pose=self.Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=self.IKContinuousMode.CONTINUOUS,
            constrained_mode=self.IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=self.FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=self.FloatValue(value=0.05),
            order_id=self.Int32Value(value=5),
        )
        robot_arm._stub.SendArmCartesianGoal(request)

    def move_gripper(
        self, arm: str, with_joint_command: bool = True, command: Optional[float] = None
    ) -> None:
        """Send the command to move the gripper of the specified arm.

        Args:
            - arm (str): The specified arm. Can be "r_arm" or "l_arm".
            - with_joint_command (bool): If True, the command is sent as a joint command.
                If False, the command is sent as an open/close command.
            - command (Optional[float]): The command to send to the gripper.
                If with_joint_command is True, this should be a float value between 0 and 100.
                If with_joint_command is False, this should be 0 (close) or 1 (open).
        """
        if arm == "l_arm":
            if not self.mirror_mode:
                gripper = self.reachy.l_arm.gripper
            else:
                gripper = self.reachy.r_arm.gripper
        else:
            if not self.mirror_mode:
                gripper = self.reachy.r_arm.gripper
            else:
                gripper = self.reachy.l_arm.gripper

        if with_joint_command:
            gripper.goal_position = command
            gripper.send_goal_positions()
        else:
            if command == 0 and not gripper.is_moving() and gripper.opening > 20:
                gripper.close()
            elif command == 1 and not gripper.is_moving() and gripper.opening < 90:
                gripper.open()

    def move_mobile_base(self, x: float, y: float, theta: float) -> None:
        """Send the command to move the mobile base.

        Args:
            - x (float): The target x position.
            - y (float): The target y position.
            - theta (float): The target orientation in degrees.
        """
        self.reachy.mobile_base.set_goal_speed(vx=x, vy=y, vtheta=theta)
        self.reachy.mobile_base.send_speed_command()

    def move_antenna(self, position: float, arm=None) -> None:
        """Send the command to move the antenna, according to the controller mode.

        Args:
            - position (float): The target position in degrees.
            - arm (Optional[str]): The side of the controller that controls the antenna.
                Can be "r_arm" or "l_arm". If None, both antennas are moved.
        """
        if arm is None:
            self.reachy.head.r_antenna.goal_position = position
            self.reachy.head.l_antenna.goal_position = -position
        elif arm == "r_arm":
            self.reachy.head.r_antenna.goal_position = position
        else:
            self.reachy.head.l_antenna.goal_position = position
        self.reachy.send_goal_positions()

    def move_head(self, orientation_matrix: np.ndarray) -> None:
        """Send the command to move the head.

        Args:
            - orientation_matrix (np.ndarray): The target orientation as a 3x3 rotation matrix.
        """
        if self.mirror_mode:
            orientation_matrix = self.get_mirror_rotation(orientation_matrix)
        orientation = R.from_matrix(orientation_matrix).as_euler("xyz", degrees=False)

        orientation = limit_orbita3d_joints(orientation, np.deg2rad(42))
        orientation = np.rad2deg(orientation)

        self.reachy.head.neck.roll.goal_position = orientation[0]
        self.reachy.head.neck.pitch.goal_position = orientation[1]
        self.reachy.head.neck.yaw.goal_position = orientation[2]
        self.reachy.head.send_goal_positions()

    def get_mirror_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Returns the symetrical rotation of the given rotation matrix.

        Args:
            rotation (np.ndarray): The input rotation matrix.
        Returns:
            np.ndarray: The mirror rotation matrix.
        """
        M = np.diag([1, -1, 1])
        return M @ rotation @ M

    def get_mirror_pose(self, pose: np.ndarray) -> np.ndarray:
        """Returns the symetrical pose of the given pose.
        Args:
            pose (np.ndarray): The input pose as a 4x4 transformation matrix.
        Returns:
            np.ndarray: The mirror pose as a 4x4 transformation matrix.
        """
        new_pose = np.copy(pose)
        new_pose[1, 3] = -new_pose[1, 3]
        new_pose[:3, :3] = self.get_mirror_rotation(new_pose[:3, :3])
        return new_pose

    def stop(self) -> None:
        """Stops the robot and turns off its components."""
        self.reachy.turn_off_smoothly()
