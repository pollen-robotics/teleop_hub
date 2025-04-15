import cv2  # type: ignore
import numpy as np

from filter.median_filter import MedianFilter


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
