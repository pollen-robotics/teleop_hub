import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore
from utils import (  # type: ignore
    load_config,
    make_homogenous_matrix_from_rotation_matrix,
)
import time
from controller.controller import Controller  # type: ignore
from controller.feetech_rustypot import FeetechSOArm  # type: ignore


class SoArmController(Controller):

    def __init__(self, port):
        super().__init__()
        self.tracker_type = "SO_arm"
        self.so_arm = FeetechSOArm(port, [1, 2, 3, 4, 5, 6])

        self.so_init_joints = {
            "r_arm": np.deg2rad([-58.33, -7.16, 27.74, 72.48, -61.67, -1.8]),
            "l_arm": np.deg2rad([-58.33, -7.16, 27.74, 72.48, -61.67, -1.8]),
            "head": np.deg2rad([-50.0, 0.0, 27.74, 72.48, -65, -41.05]),
            "mobile_base": np.deg2rad([-58.33, -7.16, 27.74, 72.48, -61.67, -1.8]),
        }

        self.so_previous_joints = {
            "r_arm": self.so_init_joints["r_arm"],
            "l_arm": self.so_init_joints["l_arm"],
            "head": self.so_init_joints["head"],
            "mobile_base": self.so_init_joints["mobile_base"],
        }


    def init_controller(self, part):
        self.so_arm.enable_torque()
        # self.so_arm.set_torque_limit(1000)
        self.so_arm.goto_joints(self.so_previous_joints[part], 2)
        time.sleep(2)
        self.so_arm.disable_torque()

    def so_arm_fk(self, joints):   
        # joints = np.deg2rad(joints)

        p1 = np.array([0, 0, 0])
        p2 = np.array([0.031, 0.0, 0.072])
        d3 = np.array([0.03, 0.0, 0.115])
        d4 = np.array([0.135, 0.0, 0.005])
        d4_bis = np.array([0.06, 0, 0])
        d5 = np.array([0.07, 0.0, 0.0])

        P1 = p1
        M1 = R.from_euler("xyz", [0, 0, -joints[0]], degrees=False).as_matrix()
        T1 = make_homogenous_matrix_from_rotation_matrix(M1, p1)
        T = T1
        P2 = T @ np.append(p2, 1)
        M2 = R.from_euler("xyz", [0, joints[1], 0], degrees=False).as_matrix()
        T2 = make_homogenous_matrix_from_rotation_matrix(M2, p2)
        T = T @ T2
        P3 = T @ np.append(d3, 1)
        M3 = R.from_euler("xyz", [0, joints[2], 0], degrees=False).as_matrix()
        T3 = make_homogenous_matrix_from_rotation_matrix(M3, d3)
        T = T @ T3
        P4 = T @ np.append(d4, 1)
        M4 = R.from_euler("xyz", [0, joints[3], 0], degrees=False).as_matrix()
        T4 = make_homogenous_matrix_from_rotation_matrix(M4, d4)
        T = T @ T4
        if len(joints) == 7:
            P4_bis = T @ np.append(d4_bis, 1)
            M4_bis = R.from_euler("xyz", [0, 0, joints[6]], degrees=False).as_matrix()
            T4_bis = make_homogenous_matrix_from_rotation_matrix(M4_bis, d4_bis)
            T = T @ T4_bis
        P5 = T @ np.append(d5, 1)
        M5 = R.from_euler("xyz", [-joints[4], 0, 0], degrees=False).as_matrix()
        T5 = make_homogenous_matrix_from_rotation_matrix(M5, d5)
        T = T @ T5
        points = np.array([P1, P2[:3], P3[:3], P4[:3], P5[:3]])
        orientation = T[:3, :3]
        # rot1 = R.from_euler('xyz', [0, 0, -np.pi/4], degrees=False).as_matrix()
        # rot2 = R.from_euler('xyz', [0, np.pi/2, 0], degrees=False).as_matrix()
        # if top_grasp:
        #     orientation = orientation @ rot2 @ rot1

        pose = make_homogenous_matrix_from_rotation_matrix(orientation, P5[:3])

        return pose
    
    def get_head_joints(self):
        """Get the head joints."""
        joints = self.so_arm.get_joints()
        return [joints[0], joints[1], joints[4]]
    
    def get_controller_pose(self):
        """Get the pose of the controller in the robot frame."""
        joints = self.so_arm.get_joints()
        pose = self.so_arm_fk(joints)
        return pose
    
    def update_joints(self, part):
        """Update the joints of the robot."""
        joints = self.so_arm.get_joints()
        self.so_previous_joints[part] = joints

    def get_gripper_joint(self):
        """Get the gripper joint."""
        joint = self.so_arm.get_joints()[5]
        return joint

    def convert_pose(self, pose):
        """Convert the pose of the tracker in the robot frame."""
        pass


    def lock_arm(self):
        """Lock the arm."""
        self.so_arm.enable_torque()
    
    def unlock_arm(self):
        """Unlock the arm."""
        self.so_arm.disable_torque()


if __name__ == "__main__":
    controller = SoArmController()