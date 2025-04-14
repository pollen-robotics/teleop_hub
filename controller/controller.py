import time

import numpy as np

from controller.aruco_cube import ArucoCube
from controller.camera import Camera
from controller.pose_filter import PoseFilter


class Controller:
    """Controller class for the robot.
    This class is responsible for managing the camera and the cube.
    It uses the ArUco markers to detect the cube and estimate its pose.
    """

    def __init__(self, arm: str, marker_size: float, camera: Camera) -> None:
        """Initialize the controller, with the camera and the corresponding ArUco cube.

        Args:
            arm (str): Corresponding arm for teleoperation. Either "l_arm" or "r_arm".
            marker_size (float): Size of the markers in meters.
            camera (Camera): Camera object to capture frames.
        """
        self.camera = camera
        self.aruco_cube = ArucoCube(self.camera, marker_size, arm)
        self.arm = arm

        self.cube_init_pose = None
        self.capture_init_pose()
        self.pose_filter = PoseFilter(alpha=0.5)

    def capture_init_pose(self) -> None:
        """Capture the initial pose of the cube."""
        while self.aruco_cube.cube_pose is None:
            cube_pose = self.aruco_cube.update_cube_pose()
            time.sleep(0.01)
        self.cube_init_pose = cube_pose

    def get_controller_pose(self) -> np.ndarray:
        """Get the pose of the cube in the robot frame.

        This function updates the cube pose using the camera frame and applies a transform to it,
        to get the pose in the robot frame. It also applies a filter to the pose to smooth it out.

        Returns:
            pose (np.ndarray): The pose of the cube in the camera frame.
        """
        pose = self.aruco_cube.update_cube_pose()
        pose_filtered = self.pose_filter.update(pose)
        pose = self.rotate_pose(pose_filtered)
        return pose

    def rotate_pose(self, pose: np.ndarray) -> np.ndarray:
        """Transform the relative pose between the initial and the current pose,
        to switch from the image frame to the robot one.

        Args:
            pose (np.ndarray): The pose of the cube in the camera frame.

        Returns:
            relative_pose (np.ndarray): The relative pose of the cube between the initial and the current one,
            in the robot frame.
        """
        if self.cube_init_pose is None:
            return pose
        relative_pose = np.linalg.inv(self.cube_init_pose) @ pose
        T_cam_to_reachy = np.array(
            [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
        )

        relative_pose = T_cam_to_reachy @ relative_pose

        return relative_pose

    def stop(self) -> None:
        """Stop the camera and close all windows."""
        self.aruco_cube.stop()
