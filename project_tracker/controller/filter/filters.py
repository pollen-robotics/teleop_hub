from collections import deque

import cv2  # type: ignore
import numpy as np


class KalmanFilter3D:
    """
    A 3D Kalman filter for smoothing 3D position measurements.
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
    ):
        self.kf = cv2.KalmanFilter(9, 3)  # 9 states : position (x,y,z) + velocity (vx,vy,vz) + acceleration (ax,ay,az)

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

    def update(self, measurement):
        """
        Update the position with a new measurement (x, y, z).
        Returns the smoothed position.
        """
        if measurement is None:
            return self.kf.statePost[:3].flatten() if self.initialized else None

        measurement = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)

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

    def __init__(self, filter_size=5):
        self.filter_size = filter_size
        self.measurements = deque(maxlen=filter_size)

    def update(self, measurement):
        self.measurements.append(measurement)
        if len(self.measurements) == self.filter_size:
            return np.median(np.array(self.measurements), axis=0)
        else:
            return measurement


class RotationSmoother:
    def __init__(self, alpha=0.3):
        """
        Initialize a filter based on an exponential moving average of rotation matrices.

        Args:
            - alpha : Smoothing coefficient between 0 (very smooth) and 1 (no smoothing).
        """

        self.alpha = alpha
        self.last_rotation_matrix = None

    def update(self, current_rotation_matrix):
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
        smoothed_rotation_matrix = (1 - self.alpha) * self.last_rotation_matrix + self.alpha * current_rotation_matrix

        # Orthogonalization of the rotation matrix
        U, _, Vt = np.linalg.svd(smoothed_rotation_matrix)
        smoothed_rotation_matrix = U @ Vt

        # Update
        self.last_rotation_matrix = smoothed_rotation_matrix
        return smoothed_rotation_matrix
