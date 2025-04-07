import time

import numpy as np

from controller.aruco_cube import ArucoCube
from controller.camera import Camera
from controller.pose_filter import PoseFilter


class Controller:
    def __init__(self, arm: str, marker_size: float, camera: Camera):
        self.camera = camera
        self.aruco_cube = ArucoCube(self.camera, marker_size, arm)
        self.arm = arm

        self.cube_init_pose = None
        self.capture_init_pose()
        self.pose_filter = PoseFilter(alpha=0.5)

    def capture_init_pose(self):
        while self.aruco_cube.cube_pose is None:
            cube_pose = self.aruco_cube.update_cube_pose()
            time.sleep(0.01)
        self.cube_init_pose = cube_pose

    def get_controller_pose(self):
        pose = self.aruco_cube.update_cube_pose()
        # pose = self.rotate_pose(pose, self.arm)
        pose_filtered = self.pose_filter.update(pose)
        pose = self.rotate_pose(pose_filtered, self.arm)
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

    def stop(self):
        self.aruco_cube.stop()
