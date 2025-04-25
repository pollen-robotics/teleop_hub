import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2  # type: ignore
import numpy as np
from camera.camera import Camera  # type: ignore
from camera.orbbec import Orbbec  # type: ignore
from camera.rgb_camera import RGBCamera  # type: ignore
from controller.joystick_controller import JoystickController  # type: ignore
from controller.rgbd_controller import ArmRGBDController, HeadRGBDController
from robots.reachy import Reachy2
from trackers.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from trackers.tracker import TrackerType  # type: ignore
from utils import load_config

DUAL_ARM = "dual_arm"
LEFT_ARM = "l_arm"
RIGHT_ARM = "r_arm"


class Teleoperation(ABC):
    def __init__(self):
        config = load_config("config.yaml")
        robot_ip = config.get("robot_ip", "localhost")
        mirror_mode = config.get("mirror_mode", False)
        self.control_mode = config.get("control_mode", "dual_arm")

        self.robot = Reachy2(robot_ip, mirror_mode)
        self.controllers = {}
        self.controller_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}
        self.robot_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}

    @abstractmethod
    def init_teleoperation(self):
        pass

    def controller_pose_to_robot_pose(self, controller_pose, arm):
        if self.differential_mode:
            return self.compute_differential_arm_pose(controller_pose, arm)
        else:
            return controller_pose.copy()

    def controller_to_head_orientation(self, controller_pose, arm):
        if self.differential_mode:
            return self.compute_differential_head_pose(controller_pose, arm)
        else:
            return controller_pose.copy()

    def compute_differential_head_pose(self, controller_pose, arm):
        diff_orientation = controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        head_orientation = diff_orientation @ self.head_previous_pose
        return head_orientation

    def teleoperation(self):
        self.init_teleoperation()
        frequency = 100
        while True:
            t = time.time()
            self.step()
            time.sleep(max(0, 1 / frequency - (time.time() - t)))

    def step(self):
        raise NotImplementedError("Each subclass must implement its own `step()` method.")


class TeleoperationRGBD(Teleoperation):
    def __init__(self):
        super().__init__()
        self.camera = Orbbec()
        self.computer_vision = ComputerVision(self.camera)
        self.controllers = {
            LEFT_ARM: ArmRGBDController(self.computer_vision, LEFT_ARM),
            RIGHT_ARM: ArmRGBDController(self.computer_vision, RIGHT_ARM),
        }
        self.head_controller = HeadRGBDController(self.computer_vision)
        self.first_command_ok = False

    def init_teleoperation(self):
        # self.robot.unfreeze(RIGHT_ARM)
        self.robot.init_robot()
        while not self.first_command_ok:
            self.computer_vision.update_landmarks_coordinates(True)
            l_pose = self.controllers[LEFT_ARM].get_controller_pose()
            r_pose = self.controllers[RIGHT_ARM].get_controller_pose()
            if self.is_first_command_ok(l_pose, False) and self.is_first_command_ok(r_pose, True):
                self.first_command_ok = True
            else:
                print(f"l_pose: {l_pose[:3,3]}, r_pose: {r_pose[:3,3]}")
                time.sleep(0.05)

    def is_first_command_ok(self, pose, is_for_r_arm):
        if pose is None:
            return False
        # check if the first command is in the cube : x 0,2/O,35 y 0,15/0.3 z -0,35/-0.2
        command = pose[:3, 3]
        if is_for_r_arm:
            command[1] = -command[1]
        if (
            command[0] < 0.35
            and command[0] > 0.2
            and command[1] < 0.3
            and command[1] > 0.15
            and command[2] < -0.15
            and command[2] > -0.35
        ):
            return True
        return False

    def step(self):
        if self.head_controller.stop_flag:
            for controller in self.controllers.values():
                controller.stop()
            self.robot.stop()
            return

        self.computer_vision.update_landmarks_coordinates(True)

        for controller in self.controllers.values():
            pose = controller.get_controller_pose()
            self.robot.go_to_pose(pose, controller.arm)
            self.controller_previous_pose[controller.arm] = pose
            gripper_command = controller.get_gripper_command()
            if gripper_command is not None:
                self.robot.move_gripper(controller.arm, False, gripper_command)
        head_pose = self.head_controller.get_controller_pose()
        self.robot.move_head(head_pose)

        rpy = self.head_controller.former_rpy[-1]

        color_frame = self.computer_vision.camera.color_frame[0]
        frame = self.computer_vision.visualization_landmarks(
            color_frame,
            self.controller_previous_pose[LEFT_ARM],
            self.controller_previous_pose[RIGHT_ARM],
            rpy[0],
            rpy[1],
            rpy[2],
            text_on=True,
        )
        cv2.imshow("Color Viewer", frame)
        cv2.waitKey(1)


class JoystickTeleoperation(Teleoperation):
    def __init__(self, tracker_type: TrackerType):
        super().__init__()
        self.tracker_type = tracker_type
        self.mode = 0
        self.camera: Optional[Camera]

        if tracker_type == TrackerType.ARUCO:
            self.camera = RGBCamera()
            self.stop = False
        elif tracker_type == TrackerType.VIVE:
            self.camera = None
            self.stop = True

        self.controllers = {
            LEFT_ARM: JoystickController(tracker_type, LEFT_ARM, self.camera),
            RIGHT_ARM: JoystickController(tracker_type, RIGHT_ARM, self.camera),
        }

        self.head_previous_pose = np.eye(3)
        self.antenna_previous_position = {RIGHT_ARM: 0, LEFT_ARM: 0}

        self.stop_count = 0

        self.joystick_mode = {LEFT_ARM: 0, RIGHT_ARM: 0}
        self.controller_previous_button = {RIGHT_ARM: [1, 1, 1], LEFT_ARM: [1, 1, 1]}

    def init_teleoperation(self):
        self.robot.unfreeze(RIGHT_ARM)
        self.robot.init_robot()
        for controller in self.controllers.values():
            controller.init_controller()
            self.controller_previous_pose[controller.arm] = controller.get_controller_pose()

            self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)

    def controller_pose_to_robot_pose(self, controller_pose, arm):
        robot_pose = self.robot_previous_pose[arm].copy()
        diff_position = controller_pose[:3, 3] - self.controller_previous_pose[arm][:3, 3]
        robot_pose[:3, 3] += diff_position
        diff_orientation = controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        robot_pose[:3, :3] = diff_orientation @ self.robot_previous_pose[arm][:3, :3]
        return robot_pose

    def controller_to_head_orientation(self, controller_pose, arm):
        diff_orientation = controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        head_orientation = diff_orientation @ self.head_previous_pose
        return head_orientation

    def joystick_to_mobile_base(self, x_input, y_input):
        command_max = 0.4
        center_joystick = 512
        x_goal, y_goal = 0, 0

        if not np.isclose(x_input, center_joystick, atol=70):
            y_goal = command_max / center_joystick * x_input - command_max
        if not np.isclose(y_input, center_joystick, atol=70):
            x_goal = command_max / center_joystick * y_input - command_max
        return x_goal, y_goal

    def trigger_joint_to_gripper_joint(self, joint, arm):
        if arm == "r_arm":
            min_joint, max_joint = -60, -15
        else:
            min_joint, max_joint = 60, 15
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper
        return gripper_opening

    def joystick_to_antenna(self, input, arm):
        center_joystick = 512
        command_max = 2

        if not np.isclose(input, center_joystick, atol=70):
            delta = command_max / center_joystick * input - command_max
            antenna_position = self.antenna_previous_position[arm] + delta
        else:
            antenna_position = self.antenna_previous_position[arm]
        return antenna_position

    def manage_mode(self):
        if self.control_mode == DUAL_ARM:
            self._handle_stop_button(self.controllers[LEFT_ARM])
            self._handle_mode_toggle(self.controllers[RIGHT_ARM])
            self._handle_mode_set(self.controllers[LEFT_ARM])
            self._handle_joystick_toggle(self.controllers[RIGHT_ARM])
            self._handle_joystick_toggle(self.controllers[LEFT_ARM])

            for controller in self.controllers.values():
                self._update_previous_buttons(controller)

        elif self.control_mode in [LEFT_ARM, RIGHT_ARM]:
            controller = self.controllers[self.control_mode]
            self._handle_stop_button(controller)
            self._handle_mode_toggle(controller)
            self._handle_joystick_toggle(controller, single_arm=True)
            self._update_previous_buttons(controller)

    def _handle_stop_button(self, controller):
        if controller.buttonB == 0:
            self.stop_count += 1
        else:
            self.stop_count = 0

        if self.stop_count == 100:
            self.stop = not self.stop
            if not self.stop:
                self.init_teleoperation()
            else:
                self.robot.stop()

    def _handle_mode_toggle(self, controller):
        if controller.buttonA == 0 and controller.buttonA != self.controller_previous_button[controller.arm][1]:
            print("Change mode")
            self.mode = 0 if self.mode == 1 else 1

    def _handle_mode_set(self, controller):
        if controller.buttonA == 0:
            self.mode = 2
        elif self.mode != 1:
            self.mode = 0

    def _handle_joystick_toggle(self, controller, single_arm=False):
        if (
            controller.joystick_button == 0
            and controller.joystick_button != self.controller_previous_button[controller.arm][0]
        ):
            print("Change joystick mode")
            if single_arm:
                self.robot.unfreeze(controller.arm)
                self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)

            self.joystick_mode[controller.arm] = 0 if self.joystick_mode[controller.arm] == 1 else 1

    def _update_previous_buttons(self, controller):
        self.controller_previous_button[controller.arm] = [
            controller.joystick_button,
            controller.buttonA,
            controller.buttonB,
        ]

    def step(self):
        # self.manage_mode()
        if not self.stop:
            self.update_robot_state()
            if tracker_type == TrackerType.ARUCO:
                frame = self.get_video_streaming()
                cv2.imshow("Color Viewer", frame)
                cv2.waitKey(1)

    def update_robot_state(self):
        if self.control_mode == DUAL_ARM:
            self.update_dual_arm_state()
        elif self.control_mode in [RIGHT_ARM, LEFT_ARM]:
            self.update_single_arm_state()
        else:
            raise ValueError(f"Invalid mode: {self.control_mode}")

    def update_dual_arm_state(self):
        if self.mode == 2:
            for controller in self.controllers.values():
                pose = controller.get_controller_pose()
                self.controller_previous_pose[controller.arm] = pose
                self.robot.unfreeze(controller.arm)
                self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)
        else:
            for controller in self.controllers.values():
                pose = controller.get_controller_pose()
                if controller.arm == RIGHT_ARM and self.mode == 1:
                    head_orientation = self.controller_to_head_orientation(pose, controller.arm)
                    self.robot.move_head(head_orientation)
                    self.head_previous_pose = head_orientation
                else:
                    robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm)
                    self.robot.go_to_pose(robot_pose, controller.arm)
                    self.robot_previous_pose[controller.arm] = robot_pose
                self.controller_previous_pose[controller.arm] = pose

            # x, y, theta = 0, 0, 0
            # if self.joystick_mode[RIGHT_ARM] == 0:
            #     _, r_y = self.joystick_to_mobile_base(
            #         self.controllers[RIGHT_ARM].joystick_x,
            #         self.controllers[RIGHT_ARM].joystick_y,
            #     )
            #     theta = r_y * 100

            # else:
            #     antenna = self.joystick_to_antenna(self.controllers[RIGHT_ARM].joystick_x, RIGHT_ARM)
            #     self.robot.move_antenna(antenna, RIGHT_ARM)
            #     self.antenna_previous_position[RIGHT_ARM] = antenna
            # if self.joystick_mode[LEFT_ARM] == 0:
            #     x, y = self.joystick_to_mobile_base(
            #         self.controllers[LEFT_ARM].joystick_x,
            #         self.controllers[LEFT_ARM].joystick_y,
            #     )
            # else:
            #     antenna = self.joystick_to_antenna(self.controllers[LEFT_ARM].joystick_x, LEFT_ARM)
            #     self.robot.move_antenna(antenna, LEFT_ARM)
            #     self.antenna_previous_position[LEFT_ARM] = antenna
            # self.robot.move_mobile_base(x, y, theta)

    def update_single_arm_state(self):
        controller = self.controllers[self.mode]
        pose = controller.get_controller_pose()
        if self.mode == 0:
            robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm)
            self.robot.go_to_pose(robot_pose, controller.arm)
            self.robot_previous_pose[controller.arm] = robot_pose
            x, y, theta = 0, 0, 0
            joystick_x, joystick_y = self.joystick_to_mobile_base(controller.joystick_x, controller.joystick_y)
            if self.joystick_mode[self.mode] == 0:
                x, y = joystick_x, joystick_y
            else:
                x = joystick_x
                theta = joystick_y * 100
            self.robot.move_mobile_base(x, y, theta)
        elif self.mode == 1:
            head_orientation = self.controller_to_head_orientation(pose, controller.arm)
            self.robot.move_head(head_orientation)
            self.head_previous_pose = head_orientation
            antenna = self.joystick_to_antenna(controller.joystick_x, controller.arm)
            self.robot.move_antenna(antenna)
            self.antenna_previous_position[controller.arm] = antenna
        self.controller_previous_pose[controller.arm] = pose

    def get_video_streaming(self):
        l_pose = self.controllers[LEFT_ARM].tracker.tracker_pose
        r_pose = self.controllers[RIGHT_ARM].tracker.tracker_pose
        frame = self.camera.get_frame_with_cube_pose([l_pose, r_pose])
        return frame


if __name__ == "__main__":

    config = load_config("config.yaml")
    tracker_type_str = config.get("tracker_type", TrackerType.ARUCO).upper()
    tracker_type = getattr(TrackerType, tracker_type_str, None)

    teleoperation: Teleoperation

    if tracker_type == TrackerType.ARUCO or tracker_type == TrackerType.VIVE:
        teleoperation = JoystickTeleoperation(tracker_type)
    elif tracker_type == TrackerType.RGBD:
        teleoperation = TeleoperationRGBD()
    else:
        raise ValueError(f"Invalid tracker type: {tracker_type}")

    try:
        teleoperation.teleoperation()

    except KeyboardInterrupt:
        for controller in teleoperation.controllers.values():
            controller.stop()
        teleoperation.robot.stop()
