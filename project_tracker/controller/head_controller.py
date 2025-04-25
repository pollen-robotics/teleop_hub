from collections import deque
from typing import Deque

import numpy as np
from controller.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from controller.rgbd_tracker.head_tracker import HeadTracker  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore


class HeadController:
    def __init__(self, computer_vision: ComputerVision, mirror_mode: bool = False) -> None:
        self.mirror_mode = mirror_mode
        self.computerVision = computer_vision
        self.tracker = HeadTracker(self.computerVision)
        self.former_rpy: Deque = deque(maxlen=10)
        self.stop_flag = False

    def get_controller_pose(self):
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose
        pose = self.convert_pose(pose)
        rpy = R.from_matrix(pose).as_euler("XYZ", degrees=True)
        self.former_rpy.append(rpy)

        return pose

    def convert_pose(self, pose):
        corrected_pose = self.computerVision.T_world_camera @ pose
        T_cam_to_reachy = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        rotation = T_cam_to_reachy @ corrected_pose @ T_cam_to_reachy.T
        return rotation

    def raise_flag_to_stop(self):
        if len(self.former_rpy) == 10:
            pitch_values = [rpy[1] for rpy in self.robot_controller.head_rpy]
            roll_values = [rpy[0] for rpy in self.robot_controller.head_rpy]
            yaw_values = [rpy[2] for rpy in self.robot_controller.head_rpy]
            if (
                np.all(np.abs(pitch_values) > 25)
                and np.all(np.abs(roll_values) < 10)
                and np.all(np.abs(yaw_values) < 10)
            ):
                print("Command to stop")
                return True
        return False
