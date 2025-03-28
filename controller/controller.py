import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.aruco_cube import ArucoCube
from controller.camera import Camera


class PoseFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.filtered_position = None
        self.filtered_rotation_quat = None

    def update(self, pose):
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]

        # --- Position ---
        if self.filtered_position is None:
            self.filtered_position = position.copy()
        else:
            self.filtered_position = (
                self.alpha * position + (1 - self.alpha) * self.filtered_position
            )

        # --- Rotation ---
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]
        if self.filtered_rotation_quat is None:
            self.filtered_rotation_quat = quat.copy()
        else:
            self.filtered_rotation_quat = (
                self.alpha * quat + (1 - self.alpha) * self.filtered_rotation_quat
            )
            self.filtered_rotation_quat /= np.linalg.norm(self.filtered_rotation_quat)

        new_pose = np.eye(4)
        new_pose[:3, 3] = self.filtered_position
        new_pose[:3, :3] = R.from_quat(self.filtered_rotation_quat).as_matrix()

        return new_pose


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
