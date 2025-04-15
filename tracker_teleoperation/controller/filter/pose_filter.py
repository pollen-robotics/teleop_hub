import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


class PoseFilter:
    """
    A simple low-pass filter for 6D poses (position and rotation).
    This filter smooths the input pose by applying a low-pass filter to both
    the position and the rotation. The filter uses an exponential moving average
    for the position and a quaternion-based low-pass filter for the rotation.
    """

    def __init__(self, alpha=0.2):
        """
        Initializes the PoseFilter with a smoothing factor.

        Args:
            alpha (float): The smoothing factor for the low-pass filter.
                A value between 0 and 1. A smaller value results in more smoothing.
        """
        self.alpha = alpha
        self.filtered_position = None
        self.filtered_rotation_quat = None

    def update(self, pose: np.ndarray) -> np.ndarray:
        """
        Updates the filter with a new pose and returns the filtered pose.

        The input pose should be a 4x4 transformation matrix.
        The output pose will also be a 4x4 transformation matrix.

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
