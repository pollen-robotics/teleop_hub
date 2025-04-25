import threading
import time
from typing import Optional

import numpy as np
from controller.arduino import ArduinoController  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from utils import make_homogenous_matrix_from_rotation_matrix  # type: ignore

from tracker_teleoperation.controller.aruco_tracker import Camera  # type: ignore
from tracker_teleoperation.controller.aruco_tracker.aruco_cube import (
    ArucoCube,  # type: ignore
)
from tracker_teleoperation.controller.filter.pose_filter import (
    PoseFilter,  # type: ignore
)
from tracker_teleoperation.controller.tracker import (  # type: ignore
    Tracker,
    TrackerType,
)
from tracker_teleoperation.controller.vive_tracker.vive_tracker import (
    ViveTracker,  # type: ignore
)

feetech_ports = {
    "l_arm": "/dev/noVR_left_motor",
    "r_arm": "/dev/noVR_right_motor",
}

arduino_ports = {
    "l_arm": "/dev/noVR_left_arduino",
    "r_arm": "/dev/noVR_right_arduino",
}

gripper_joints = {"l_arm": [65, 30], "r_arm": [-65, -30]}


class Controller:
    def __init__(
        self,
        tracker_type: TrackerType,
        arm: str,
        marker_size: Optional[float],
        camera: Optional[Camera]
    ) -> None:
        self.tracker_type = tracker_type
        self.arm = arm
        self.arduino = ArduinoController(arduino_ports[arm])
        self.tracker: Tracker

        if self.tracker_type == TrackerType.ARUCO:
            self.camera = camera
            if self.camera is None:
                raise ValueError("Camera is required for ArUco tracking.")
            if marker_size is None:
                marker_size = 0.05
                print("Default marker size set to 0.05 m.")
            self.tracker = ArucoCube(arm, self.camera, marker_size)
            self.pose_filter = PoseFilter(alpha=0.5)
            self.is_filtered = True

        elif self.tracker_type == TrackerType.VIVE:
            self.tracker = ViveTracker(arm)
            self.is_filtered = False

        else:
            raise ValueError(
                f"Tracker type '{tracker_type}' not recognized."
            )

        self.stop_flag = False

        self.joystick_x = None
        self.joystick_y = None
        self.joystick_button = None
        self.buttonA = None
        self.buttonB = None

        thread = threading.Thread(target=self._update_arduino_data)
        thread.daemon = True
        thread.start()

        self.gripper_joints_limit = gripper_joints[arm]
        # self.init_gripper(self.gripper_joints_limit[1])
        # self.init_controller()

    # def init_gripper(self, joint):
    #     self.feetech.enable_torque()
    #     self.feetech.goto_joints([joint], 1)
    #     self.feetech.disable_torque()

    def init_controller(self):
        self.tracker.update_tracker_pose()
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
    #     return self.feetech.get_joints()[0]

    def get_controller_pose(self):
        pose = self.tracker.update_tracker_pose()
        if self.is_filtered:
            pose = self.pose_filter.update(pose)
        pose = self.tracker.tracker_pose
        pose = self.rotate_pose(pose)
        return pose

    def rotate_pose(self, pose):
        if self.tracker_init_pose is None:
            return pose

        # Get the relative pose between the initial and the current pose
        relative_pose = np.linalg.inv(self.tracker_init_pose) @ pose

        if self.tracker_type == TrackerType.VIVE:
            if self.arm == "r_arm":
                angle_rotation = -30
            else:
                angle_rotation = 30

            rotation = R.from_euler("xyz", [0, 0, angle_rotation], degrees=True).as_matrix()
            Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
            relative_pose = Trot @ relative_pose

            if self.arm == "r_arm":
                rotation = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
                Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
                relative_pose = Trot @ relative_pose

            if self.arm == "l_arm":
                rotation = R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
                Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
                relative_pose = Trot @ relative_pose

        elif self.tracker_type == TrackerType.ARUCO:
            T_cam_to_reachy = np.array(
                [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
            )
            relative_pose = T_cam_to_reachy @ relative_pose

        return relative_pose

    def stop(self):
        self.stop_flag = True
        self.arduino.close()
        self.tracker.stop()


if __name__ == "__main__":
    controller = Controller(TrackerType.VIVE, "r_arm", None, None)

    frequency = 100
    try:
        while True:
            print(
                (
                    f"x: {controller.joystick_x}, y: {controller.joystick_y}, "
                    f"button_cmd: {controller.joystick_button}, buttonA: {controller.buttonA}, "
                    f"buttonB: {controller.buttonB}"
                )
            )
            # gripper_joint = controller.get_gripper_joint()
            # print(f"Gripper joint: {gripper_joint}")
            pose = controller.get_controller_pose()
            print(f"Controller pose: {pose}")
            time.sleep(1 / frequency)

    except KeyboardInterrupt:
        controller.stop()
