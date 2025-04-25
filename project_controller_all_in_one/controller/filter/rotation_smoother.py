import numpy as np


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
