import time

import numpy as np
from controller.filter.filters import KalmanFilter3D, RotationSmoother  # type: ignore
from controller.rgbd_tracker.computer_vision import ComputerVision  # type: ignore


class HeadTracker:
    def __init__(self, computerVision: ComputerVision) -> None:
        self.computerVision = computerVision
        self.tracker_pose: np.ndarray | None = None

        self.kf = [KalmanFilter3D() for _ in range(4)]

        self.rotation_smoother = RotationSmoother()

    def get_head_points(self):
        face_points = self.computerVision.face_points
        return face_points

    def estimate_head_orientation(self, head_points) -> np.ndarray:
        nose = head_points[0]
        chin = head_points[1]
        left_eye = head_points[2]
        right_eye = head_points[3]

        # in IMAGE FRAME
        x_axis = np.array(right_eye) - np.array(left_eye)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.array(chin) - np.array(nose)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        rot = np.array([x_axis, y_axis, z_axis]).T

        return rot

    def update_tracker_pose(self, mirror_mode: bool = False):
        head_points = self.get_head_points()
        # Apply Kalman filter to smooth the head points
        head_points_filtered = [self.kf[i].update(head_points[i]) for i in range(4)]
        # get the rotation matrix
        tracker_pose = self.estimate_head_orientation(head_points_filtered)
        self.tracker_pose = tracker_pose
