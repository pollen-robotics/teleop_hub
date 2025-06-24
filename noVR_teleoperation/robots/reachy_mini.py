from typing import Optional

import numpy as np
# robots.robot import Robot  # type: ignore
from robots.robot import Robot  # type: ignore
from reachy_mini import ReachyMini

# from stewart_little_control import Client
from scipy.spatial.transform import Rotation as R  # type: ignore



class ReachyMiniTeleoperation(Robot):
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
        self.reachy_mini = ReachyMini(spawn_daemon=True, use_sim=True)
        # self.client = Client(robot_ip)
        self.target_pose = np.eye(4)
        self.target_antennas = np.array([0., 0.])


    def fk(self, arm) -> np.ndarray:
        """Compute the forward kinematics of the specified arm.

        Args:
            arm (str): The specified arm. Can be "r_arm" or "l_arm".
        Returns:
            np.ndarray: The effector pose of the specified arm as a 4x4 transformation matrix.
        """
        return np.eye(4)  # Placeholder for actual FK computation

    def go_to_pose(self, pose: np.ndarray, arm: str) -> None:
        """Send the command to move the specified arm to the target pose.

        Args:
            pose (np.ndarray): The target pose as a 4x4 transformation matrix.
            arm (str): The specified arm. Can be "r_arm" or "l_arm".
            
        """
        # joints = self.client.get_joint_positions()
        # print(joints)
        if arm == "r_arm":
            # self.client.send_pose(pose, offset_zero = True)
            self.target_pose[:3, :3] = pose[:3, :3]
        if arm == "l_arm":
            # self.client.send_pose(pose, offset_zero = True)
            self.target_pose[:3, 3] = pose[:3, 3]
            pass


    def limit_pose(self, pose: np.ndarray) -> np.ndarray:
        x_limit = [-0.03, 0.03]
        y_limit = [-0.03, 0.03]
        z_limit = [-0.03, 0.02]
        # Extract the translation part of the pose
        translation = pose[:3, 3]
        # Apply limits
        translation[0] = np.clip(translation[0], x_limit[0], x_limit[1])
        translation[1] = np.clip(translation[1], y_limit[0], y_limit[1])
        translation[2] = np.clip(translation[2], z_limit[0], z_limit[1])
        # Update the pose with the limited translation
        pose[:3, 3] = translation

        roll_limit = [-np.pi / 6, np.pi / 6]
        pitch_limit = [-np.pi / 8, np.pi / 8]
        yaw_limit = [-np.pi / 4, np.pi / 4]
        # Extract the rotation part of the pose
        rotation = pose[:3, :3]
        # Convert rotation matrix to Euler angles
        euler_angles = R.from_matrix(rotation).as_euler('xyz')
        # print(euler_angles)
        # Apply limits
        euler_angles[0] = np.clip(euler_angles[0], roll_limit[0], roll_limit[1])
        euler_angles[1] = np.clip(euler_angles[1], pitch_limit[0], pitch_limit[1])
        euler_angles[2] = np.clip(euler_angles[2], yaw_limit[0], yaw_limit[1])
        # Convert back to rotation matrix
        rotation = R.from_euler('xyz', euler_angles).as_matrix()
        # Update the pose with the limited rotation
        pose[:3, :3] = rotation

        return pose


    def move_antenna(self, position: float, arm=None) -> None:
        """Send the command to move the antenna, according to the controller mode.

        Args:
            - position (float): The target position in degrees.
            - arm (Optional[str]): The side of the controller that controls the antenna.
                Can be "r_arm" or "l_arm". If None, both antennas are moved.
        """
        print(position)
        if arm == "r_arm":
            self.target_antennas[0] = np.deg2rad(position)
        if arm == "l_arm":
            self.target_antennas[1] = np.deg2rad(position)
            self.limit_pose(self.target_pose)

            # self.client.send_pose(self.target_pose, antennas = self.target_antennas, offset_zero = True)
            print(self.target_antennas)
            self.reachy_mini.set_position(head=self.target_pose, antennas=self.target_antennas)
            # self.limit_pose(self.target_pose)
            # print(self.target_pose)


    def move_gripper(self, arm, with_joint_command = True, command = None):
        # if arm == "r_arm":
        #     self.target_antennas[0] = np.deg2rad(command)
        # if arm == "l_arm":
        #     self.target_antennas[1] = np.deg2rad(-command)
        #     self.limit_pose(self.target_pose)
            # print(self.target_pose)
        pass


    def move_mobile_base(self, x: float, y: float, theta: float) -> None:
        pass

    def move_head(self, head_joints: np.ndarray) -> None:
        pass

    def stop(self):
        pass

    def init_robot(self):
        pass

    def unfreeze(self, arm):
        pass
