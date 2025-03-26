from aruco_cube import ArucoCube
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from utils import make_homogenous_matrix_from_rotation_matrix


class Controller:
    def __init__(self, arm):
        self.aruco_cube = ArucoCube()
        self.arm = arm

        self.cube_init_pose = None
        self.capture_init_pose()

    def capture_init_pose(self):
        while self.aruco_cube.cube_pose is None:
            self.aruco_cube.update_cube_pose()
            time.sleep(0.01)
        self.cube_init_pose = self.aruco_cube.cube_pose

    def get_controller_pose(self):
        self.aruco_cube.update_cube_pose()
        pose = self.aruco_cube.cube_pose
        pose = self.rotate_pose(pose, self.arm)
        return pose


    def rotate_pose(self, pose, arm):
        relative_pose = np.linalg.inv(self.cube_init_pose) @ pose
        T_cam_to_reachy = np.array([
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        relative_pose = T_cam_to_reachy @ relative_pose

        return relative_pose
        


