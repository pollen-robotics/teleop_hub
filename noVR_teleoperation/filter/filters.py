from collections import deque
from typing import Deque, Optional

import cv2  # type: ignore
import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


class KalmanFilter3D:
    """A 3D Kalman filter for smoothing 3D position measurements.

    This filter uses a constant velocity model and can be used to smooth noisy measurements of 3D positions (x, y, z).
    The state vector is defined as [x, y, z, vx, vy, vz, ax, ay, az]
    where (x, y, z) are the positions, (vx, vy, vz) are the velocities, and (ax, ay, az) are the accelerations.
    """

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=0.1,
        error_cov_post=1.0,
        add_median_filter=True,
        median_filter_size=5,
        dt=1 / 20,  # Time between measurements (20 Hz)
    ) -> None:
        """Initializes the Kalman filter with the specified parameters.

        Args:
            - process_noise : Process noise covariance.
            - measurement_noise : Measurement noise covariance.
            - error_cov_post : Initial error covariance.
            - add_median_filter : Whether to add a median filter for smoothing.
            - median_filter_size : Size of the median filter window.
            - dt : Time step between measurements (in seconds).
        """
        self.kf = cv2.KalmanFilter(
            9, 3
        )  # 9 states : position (x,y,z) + velocity (vx,vy,vz) + acceleration (ax,ay,az)

        # Measurement matrix : only position (x,y,z)
        self.kf.measurementMatrix = np.zeros((3, 9), dtype=np.float32)
        self.kf.measurementMatrix[:3, :3] = np.eye(3)

        # Transition matrix : depends on position, velocity and acceleration
        self.kf.transitionMatrix = np.eye(9, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = dt  # x depends on v
            self.kf.transitionMatrix[i + 3, i + 6] = dt  # v depends on a

        # Process noise, measurement noise and error covariance
        self.kf.processNoiseCov = np.eye(9, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * error_cov_post

        self.initialized = False
        self.kf.statePre = np.zeros((9, 1), dtype=np.float32)
        self.kf.statePost = np.zeros((9, 1), dtype=np.float32)

        self.add_median_filter = add_median_filter
        if self.add_median_filter:
            self.median_filter = MedianFilter(median_filter_size)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update the position with a new measurement (x, y, z).
        This method applies the Kalman filter to the new measurement and returns the smoothed position.

        Args:
            - measurement : A 3D position measurement (x, y, z) as a numpy array.

        Returns:
            - filtered_position : The smoothed position as a numpy array.
                If no measurement is provided, the filter will return the last known position.
        """
        if measurement is None:
            return self.kf.statePost[:3].flatten() if self.initialized else None

        measurement = np.array(
            [[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32
        )

        if not self.initialized:
            self.kf.statePre[:3] = measurement
            self.kf.statePost[:3] = measurement
            self.initialized = True
            return measurement.flatten()

        self.kf.correct(measurement)
        prediction = self.kf.predict()
        filtered_position = np.array(prediction[:3]).flatten()

        if self.add_median_filter:
            filtered_position = self.median_filter.update(filtered_position)

        return filtered_position


class MedianFilter:
    """A simple median filter for smoothing data.

    This filter maintains a sliding window of the last N measurements
    and returns the median of those measurements.
    """

    def __init__(self, filter_size: int = 5) -> None:
        """Initializes the median filter with the specified window size.
        Args:
            - filter_size : Size of the sliding window for the median filter.
        """
        if filter_size < 1:
            raise ValueError("Filter size must be at least 1.")

        self.filter_size = filter_size
        self.measurements: Deque = deque(maxlen=filter_size)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update the filter with a new measurement and return the smoothed value.
        Args:
            - measurement : A new measurement to be added to the filter.
        Returns:
            - smoothed_value : The median of the last N measurements.
        """
        self.measurements.append(measurement)
        if len(self.measurements) == self.filter_size:
            return np.median(np.array(self.measurements), axis=0)
        else:
            return measurement


class RotationSmoother:
    def __init__(self, alpha: float = 0.3) -> None:
        """
        Initialize a filter based on an exponential moving average of rotation matrices.

        Args:
            - alpha : Smoothing coefficient between 0 (very smooth) and 1 (no smoothing).
        """

        self.alpha = alpha
        self.last_rotation_matrix: Optional[np.ndarray] = None

    def update(self, current_rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing on the rotation.

        Args:
            - current_rotation_matrix : 3x3 matrix representing the current orientation.

        Returns:
            - Smoothed 3x3 rotation matrix.
        """

        if self.last_rotation_matrix is None:
            self.last_rotation_matrix = current_rotation_matrix
            return current_rotation_matrix

        # Exponential Moving Average on the rotation matrix
        smoothed_rotation_matrix = (
            1 - self.alpha
        ) * self.last_rotation_matrix + self.alpha * current_rotation_matrix

        # Orthogonalization of the rotation matrix
        U, _, Vt = np.linalg.svd(smoothed_rotation_matrix)
        smoothed_rotation_matrix = U @ Vt

        # Update
        self.last_rotation_matrix = smoothed_rotation_matrix
        return smoothed_rotation_matrix


class PoseFilter:
    """A simple low-pass filter for 6D poses (position and rotation).

    This filter smooths the input pose by applying a low-pass filter to both
    the position and the rotation. The filter uses an exponential moving average
    for the position and a quaternion-based low-pass filter for the rotation.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        """
        Initializes the PoseFilter with a smoothing factor.

        Args:
            alpha (float): The smoothing factor for the low-pass filter.
                A value between 0 and 1. A smaller value results in more smoothing.
        """
        self.alpha = alpha
        self.filtered_position: Optional[np.ndarray] = None
        self.filtered_rotation_quat = None

    def update(self, pose: np.ndarray) -> np.ndarray:
        """
        Updates the filter with a new pose and returns the filtered pose.

        Args:
            pose (np.ndarray): The input pose as a 4x4 transformation matrix.

        Returns:
            np.ndarray: The filtered pose as a 4x4 transformation matrix.
        """
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
        quat = r.as_quat()
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
