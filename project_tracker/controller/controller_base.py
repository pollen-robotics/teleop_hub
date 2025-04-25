# import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional

import numpy as np
from controller.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from controller.rgbd_tracker.rgbd_tracker import (  # type: ignore
    ArmRGBDTracker,
    GripperRGBDTracker,
    HeadRGBDTracker,
)

# from controller.arduino import ArduinoController  # type: ignore
# from controller.aruco_tracker import Camera  # type: ignore
# from controller.aruco_tracker.aruco_cube import ArucoCube  # type: ignore
# from controller.aruco_tracker.computer_vision import ComputerVision  # type: ignore
# from controller.feetech import Feetech  # type: ignore
# from controller.filter.pose_filter import PoseFilter  # type: ignore
# from camera.camera import Camera  # type: ignore
from controller.tracker import Tracker, TrackerType  # type: ignore
from reachy2_sdk.utils.utils import recompose_matrix  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore

# from controller.vive_tracker.vive_tracker import ViveTracker  # type: ignore
# from utils import make_homogenous_matrix_from_rotation_matrix  # type: ignore

feetech_ports = {
    "l_arm": "/dev/noVR_left_motor",
    "r_arm": "/dev/noVR_right_motor",
}

arduino_ports = {
    "l_arm": "/dev/noVR_left_arduino",
    "r_arm": "/dev/noVR_right_arduino",
}

gripper_joints = {"l_arm": [65, 30], "r_arm": [-65, -30]}


class Controller(ABC):
    def __init__(self, mirror_mode: bool = False) -> None:
        self.mirror_mode = mirror_mode
        self.tracker: Tracker

    @abstractmethod
    def get_controller_pose(self):
        pass

    @abstractmethod
    def convert_pose(self, pose):
        pass


class RGBDController(Controller, ABC):
    def __init__(self, computer_vision: ComputerVision, mirror_mode: bool = False) -> None:
        super().__init__(mirror_mode)
        self.computer_vision = computer_vision
        self.init_user_parameters()

    def init_user_parameters(self):
        while not np.any(self.computer_vision.user_center):
            time.sleep(0.2)
        self.user_center = self.computer_vision.user_center
        real_dist_intershoulder = 0.3
        self.normalization_factor = real_dist_intershoulder / self.computer_vision.dist_intershoulder

    def get_controller_pose(self):
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose
        pose = self.convert_pose(pose)
        return pose


class HeadRGBDController(RGBDController):
    def __init__(self, computer_vision: ComputerVision, mirror_mode: bool = False) -> None:
        super().__init__(computer_vision, mirror_mode)
        self.tracker = HeadRGBDTracker(self.computer_vision)
        self.former_rpy: Deque = deque(maxlen=10)
        self.stop_flag = False

    def convert_pose(self, pose):
        corrected_pose = self.computer_vision.T_world_camera @ pose
        T_cam_to_reachy = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        rotation = T_cam_to_reachy @ corrected_pose @ T_cam_to_reachy.T
        rpy = R.from_matrix(rotation).as_euler("XYZ", degrees=True)

        self.former_rpy.append(rpy)
        if self.raise_flag_to_stop():
            print("Command to stop")
            self.stop_flag = True

        return rotation

    def raise_flag_to_stop(self):
        if len(self.former_rpy) == 10:
            pitch_values = [rpy[1] for rpy in self.former_rpy]
            roll_values = [rpy[0] for rpy in self.former_rpy]
            yaw_values = [rpy[2] for rpy in self.former_rpy]
            if (
                np.all(np.abs(pitch_values) > 25)
                and np.all(np.abs(roll_values) < 10)
                and np.all(np.abs(yaw_values) < 10)
            ):
                print("Command to stop")
                return True
        return False


class ArmRGBDController(RGBDController):
    def __init__(self, computer_vision: ComputerVision, arm: str, mirror_mode: bool = False) -> None:
        super().__init__(computer_vision, mirror_mode)
        self.arm = arm
        self.tracker = ArmRGBDTracker(self.arm, self.computer_vision)
        self.gripper = GripperRGBDTracker(self.arm, self.computer_vision)
        self.stop_flag = False

    def convert_pose(self, pose):
        corrected_rotation = self.computer_vision.T_world_camera @ pose[:3, :3]
        corrected_position = self.computer_vision.T_world_camera @ (pose[:3, 3] - self.user_center)

        normalized_position = corrected_position * self.normalization_factor

        x = normalized_position[0]
        y = normalized_position[1]
        z = normalized_position[2]

        new_position = np.array([-z, x, -y])

        T_cam_to_reachy = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        new_rotation = T_cam_to_reachy @ corrected_rotation @ T_cam_to_reachy.T

        new_pose = recompose_matrix(new_rotation, new_position)

        return new_pose

    def get_gripper_command(self):
        opening = self.gripper.update_tracker_pose()
        if opening is None:
            return None

        if opening < 1:
            command = 0
        elif opening > 1.2:
            command = 1
        else:
            command = None

        return command

    def stop(self):
        self.stop_flag = True
        self.tracker.stop()


# class JoystickController(Controller):
#     def __init__(
#         self, tracker_type: TrackerType, arm: str, camera: Optional[Camera], marker_size: Optional[float], mirror_mode: bool = False
#     ) -> None:
#         super().__init__(mirror_mode)
#         self.tracker_type = tracker_type
#         self.arm = arm
#         self.arduino = ArduinoController(arduino_ports[arm])

#         if self.tracker_type == TrackerType.ARUCO:
#             self.camera = camera
#             if self.camera is None:
#                 raise ValueError("Camera is required for ArUco tracking.")
#             if marker_size is None:
#                 marker_size = 0.05
#                 print("Default marker size set to 0.05 m.")

#             self.tracker = ArucoCube(arm, self.camera, marker_size)
#             self.pose_filter = PoseFilter(alpha=0.5)
#             self.is_filtered = True

#         elif self.tracker_type == TrackerType.VIVE:
#             self.tracker = ViveTracker(arm)
#             self.is_filtered = False

#         self.gripper = Feetech(feetech_ports[arm])
#         self.joystick_x = None
#         self.joystick_y = None
#         self.joystick_button = None
#         self.buttonA = None
#         self.buttonB = None

#         thread = threading.Thread(target=self._update_arduino_data)
#         thread.daemon = True
#         thread.start()

#         self.init_controller()
#         self.init_gripper(self.gripper_joints_limit[1])
#         self.stop_flag = False
#         self.gripper_joints_limit = gripper_joints[arm]

#     def init_gripper(self, joint):
#         self.feetech.enable_torque()
#         self.feetech.goto_joints([joint], 1)
#         self.feetech.disable_torque()

#     def init_controller(self):
#         self.tracker.update_tracker_pose()
#         self.tracker_init_pose = self.tracker.tracker_pose

#     def _update_arduino_data(self):
#         while not self.stop_flag:
#             x, y, button_cmd, buttonA, buttonB = self.arduino.read()
#             if x is not None:
#                 self.joystick_button = button_cmd
#                 if self.arm == "r_arm":
#                     self.buttonA = buttonA
#                     self.buttonB = buttonB
#                     self.joystick_x = x
#                     self.joystick_y = y
#                 else:
#                     self.buttonA = buttonB
#                     self.buttonB = buttonA
#                     self.joystick_x = 1024 - x
#                     self.joystick_y = 1024 - y
#             else:
#                 print("No data")
#             time.sleep(0.1)

#     def get_gripper_joint(self):
#         return self.gripper.get_joints()[0]

#     def get_controller_pose(self):
#         self.tracker.update_tracker_pose()
#         pose = self.tracker.tracker_pose
#         if self.is_filtered:
#             pose = self.pose_filter.update(pose)
#         pose = self.convert_pose(pose)
#         self.former_pose = pose
#         return pose

#     def convert_pose(self, pose):
#         if self.tracker_init_pose is None:
#             return pose

#         # Get the relative pose between the initial and the current pose
#         relative_pose = np.linalg.inv(self.tracker_init_pose) @ pose

#         if self.tracker_type == TrackerType.VIVE:
#             relative_pose = self.convert_for_vive_tracker(relative_pose)

#         elif self.tracker_type == TrackerType.ARUCO:
#             T_cam_to_reachy = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
#             relative_pose = T_cam_to_reachy @ relative_pose

#         return relative_pose

#     def convert_for_vive_tracker(self, pose):
#         if self.arm == "r_arm":
#             angle_rotation = -30
#         else:
#             angle_rotation = 30

#         rotation = R.from_euler("xyz", [0, 0, angle_rotation], degrees=True).as_matrix()
#         Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
#         pose = Trot @ pose

#         if self.arm == "r_arm":
#             rotation = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
#             Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
#             pose = Trot @ pose

#         if self.arm == "l_arm":
#             rotation = R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
#             Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
#             pose = Trot @ pose

#         return pose

#     def get_gripper_command(self):
#         opening = self.gripper.update_gripper_opening()
#         if opening is None:
#             return None

#         if opening < 1:
#             command = 0
#         elif opening > 1.2:
#             command = 1
#         else:
#             command = None

#         return command

#     def stop(self):
#         self.stop_flag = True
#         self.arduino.close()
#         self.tracker.stop()
