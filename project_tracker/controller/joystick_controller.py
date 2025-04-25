import threading
import time
from typing import Optional

import numpy as np
from camera.camera import Camera  # type: ignore
from controller.arduino import ArduinoController  # type: ignore
from controller.controller import Controller  # type: ignore
from controller.feetech import Feetech  # type: ignore
from filter.filters import PoseFilter  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from trackers.aruco_tracker.aruco_cube import ArucoCube  # type: ignore
from trackers.tracker import TrackerType  # type: ignore
from trackers.vive_tracker.vive_tracker import ViveTracker  # type: ignore
from utils import (  # type: ignore
    load_config,
    make_homogenous_matrix_from_rotation_matrix,
)

# feetech_ports = {
#     "l_arm": "/dev/noVR_left_motor",
#     "r_arm": "/dev/noVR_right_motor",
# }

# arduino_ports = {
#     "l_arm": "/dev/noVR_left_arduino",
#     "r_arm": "/dev/noVR_right_arduino",
# }

# gripper_joints = {"l_arm": [65, 30], "r_arm": [-65, -30]}


class JoystickController(Controller):
    def __init__(
        self,
        tracker_type: TrackerType,
        arm: str,
        camera: Optional[Camera],
    ) -> None:
        super().__init__()
        self.tracker_type = tracker_type
        self.arm = arm
        # self.arduino = ArduinoController(arduino_ports[arm])
        # self.gripper = Feetech(feetech_ports[arm])

        if self.tracker_type == TrackerType.ARUCO:
            self.camera = camera
            if self.camera is None:
                raise ValueError("Camera is required for ArUco tracking.")

            self.tracker = ArucoCube(arm, self.camera)
            self.pose_filter = PoseFilter(alpha=0.5)
            self.is_filtered = True

        elif self.tracker_type == TrackerType.VIVE:
            self.tracker = ViveTracker(arm)
            self.is_filtered = False

        # self.gripper = Feetech(feetech_ports[arm])
        # self.joystick_x = None
        # self.joystick_y = None
        # self.joystick_button = None
        # self.buttonA = None
        # self.buttonB = None

        self.stop_flag = False
        self.gripper_joints_limit = load_config("config.yaml")["gripper_joints_limit"][self.arm]

        # thread = threading.Thread(target=self._update_arduino_data)
        # thread.daemon = True
        # thread.start()

        self.init_controller()
        # self.init_gripper(self.gripper_joints_limit[1])

    # def init_gripper(self, joint):
    #     self.feetech.enable_torque()
    #     self.feetech.goto_joints([joint], 1)
    #     self.feetech.disable_torque()

    def init_controller(self):
        while self.tracker.tracker_pose is None:
            self.tracker.update_tracker_pose()
            time.sleep(0.1)
        self.tracker_init_pose = self.tracker.tracker_pose

    def _update_arduino_data(self):
        while not self.stop_flag:
            x, y, button_cmd, buttonA, buttonB = self.arduino.read()
            if x is not None:
                self.joystick_button = button_cmd
                if self.arm == "r_arm":
                    self.buttonA = buttonA
                    self.buttonB = buttonB
                    self.joystick_x = x
                    self.joystick_y = y
                else:
                    self.buttonA = buttonB
                    self.buttonB = buttonA
                    self.joystick_x = 1024 - x
                    self.joystick_y = 1024 - y
            else:
                print("No data")
            time.sleep(0.1)

    # def get_gripper_joint(self):
    #     return self.gripper.get_joints()[0]

    def get_controller_pose(self):
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose

        if self.is_filtered:
            pose = self.pose_filter.update(pose)
        pose = self.convert_pose(pose)
        self.former_pose = pose
        return pose

    def convert_pose(self, pose):
        if self.tracker_init_pose is None:
            return pose

        # Get the relative pose between the initial and the current pose
        relative_pose = np.linalg.inv(self.tracker_init_pose) @ pose

        if self.tracker_type == TrackerType.VIVE:
            relative_pose = self.convert_for_vive_tracker(relative_pose)

        elif self.tracker_type == TrackerType.ARUCO:
            T_cam_to_reachy = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            relative_pose = T_cam_to_reachy @ relative_pose

        return relative_pose

    def convert_for_vive_tracker(self, pose):
        if self.arm == "r_arm":
            angle_rotation = -30
        else:
            angle_rotation = 30

        rotation = R.from_euler("xyz", [0, 0, angle_rotation], degrees=True).as_matrix()
        Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
        pose = Trot @ pose

        if self.arm == "r_arm":
            rotation = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
            Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
            pose = Trot @ pose

        if self.arm == "l_arm":
            rotation = R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
            Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
            pose = Trot @ pose

        return pose

    # def get_gripper_command(self):
    #     opening = self.gripper.update_gripper_opening()
    #     if opening is None:
    #         return None

    #     if opening < 1:
    #         command = 0
    #     elif opening > 1.2:
    #         command = 1
    #     else:
    #         command = None

    #     return command

    def stop(self):
        self.stop_flag = True
        # self.arduino.close()
        self.tracker.stop()
