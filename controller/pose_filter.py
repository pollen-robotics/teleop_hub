import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


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
