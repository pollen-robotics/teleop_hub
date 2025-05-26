import time

import numpy as np
from controllers.controller import Controller  # type: ignore
from controllers.feetech_rustypot import Feetech  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from trackers.tracker import TrackerType  # type: ignore
from utils import make_homogenous_matrix_from_rotation_matrix  # type: ignore


class SoArmController(Controller):
    """SoArmController class.

    This class defines the interface for the S0100 controller, which uses a Feetech servo controller
    to control either one of the arms or the head of the robot.
    The trigger controls the gripper if the SO100 controls the arm, or the antenna if it controls the head.
    It is a subclass of the Controller class.
    """

    def __init__(self, port: str) -> None:
        """Initialize the SoArmController.
        Args:
            port (str): The serial port to which the SO100 is connected.
        """
        super().__init__()
        self.tracker_type = TrackerType.ARM
        self.so_arm = Feetech(port, [1, 2, 3, 4, 5, 6])

        # Initialize the SO100 controller with the initial joint positions.
        self.so_init_joints = {
            "r_arm": np.deg2rad([-58.33, -7.16, 27.74, 72.48, -61.67, -1.8]),
            "l_arm": np.deg2rad([-58.33, -7.16, 27.74, 72.48, -61.67, -1.8]),
            "head": np.deg2rad([-50.0, 0.0, 27.74, 72.48, -65, -41.05]),
            "mobile_base": np.deg2rad([-58.33, -7.16, 27.74, 72.48, -61.67, -1.8]),
        }

        # Record the previous joint positions for each part of the robot.
        self.so_previous_joints = {
            "r_arm": self.so_init_joints["r_arm"],
            "l_arm": self.so_init_joints["l_arm"],
            "head": self.so_init_joints["head"],
            "mobile_base": self.so_init_joints["mobile_base"],
        }

    def init_controller(self, part: str) -> None:
        """Initialize the controller for a specific part of the robot."""
        self.so_arm.enable_torque()
        self.so_arm.goto_joints(self.so_previous_joints[part], 2)
        time.sleep(2)
        self.so_arm.disable_torque()

    def so_arm_fk(self, joints: list[float]) -> np.ndarray:
        """Forward kinematics for the SO100 arm.

        Computes the pose of the end effector based on the joint angles.

        Args:
            joints (list[float]): List of joint angles in radians.
        Returns:
            np.ndarray: The pose of the end effector in the robot frame as a 4x4 homogeneous transformation matrix.
        """
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.031, 0.0, 0.072])
        d3 = np.array([0.03, 0.0, 0.115])
        d4 = np.array([0.135, 0.0, 0.005])
        d4_bis = np.array([0.06, 0, 0])
        d5 = np.array([0.07, 0.0, 0.0])

        M1 = R.from_euler("xyz", [0, 0, -joints[0]], degrees=False).as_matrix()
        T1 = make_homogenous_matrix_from_rotation_matrix(M1, p1)
        T = T1

        M2 = R.from_euler("xyz", [0, joints[1], 0], degrees=False).as_matrix()
        T2 = make_homogenous_matrix_from_rotation_matrix(M2, p2)
        T = T @ T2

        M3 = R.from_euler("xyz", [0, joints[2], 0], degrees=False).as_matrix()
        T3 = make_homogenous_matrix_from_rotation_matrix(M3, d3)
        T = T @ T3

        M4 = R.from_euler("xyz", [0, joints[3], 0], degrees=False).as_matrix()
        T4 = make_homogenous_matrix_from_rotation_matrix(M4, d4)
        T = T @ T4

        if len(joints) == 7:
            M4_bis = R.from_euler("xyz", [0, 0, joints[6]], degrees=False).as_matrix()
            T4_bis = make_homogenous_matrix_from_rotation_matrix(M4_bis, d4_bis)
            T = T @ T4_bis

        P5 = T @ np.append(d5, 1)
        M5 = R.from_euler("xyz", [-joints[4], 0, 0], degrees=False).as_matrix()
        T5 = make_homogenous_matrix_from_rotation_matrix(M5, d5)
        T = T @ T5

        orientation = T[:3, :3]
        pose = make_homogenous_matrix_from_rotation_matrix(orientation, P5[:3])

        return pose

    def get_head_joints(self) -> list[float]:
        """Get the head joints."""
        joints = self.so_arm.get_joints()
        return [joints[0], joints[1], joints[4]]

    def get_controller_pose(self) -> np.ndarray:
        """Get the pose of the controller in the robot frame."""
        joints = self.so_arm.get_joints()
        pose = self.so_arm_fk(joints)
        return pose

    def update_joints(self, part: str) -> None:
        """Update the joints of the robot."""
        joints = self.so_arm.get_joints()
        self.so_previous_joints[part] = joints

    def get_gripper_joint(self) -> float:
        """Get the gripper joint value."""
        joint = self.so_arm.get_joints()[5]
        return joint

    def convert_pose(self, pose: np.ndarray):
        """Convert the pose of the tracker in the robot frame."""
        pass

    def lock_arm(self) -> None:
        """Lock the arm."""
        self.so_arm.enable_torque()

    def unlock_arm(self) -> None:
        """Unlock the arm."""
        self.so_arm.disable_torque()
