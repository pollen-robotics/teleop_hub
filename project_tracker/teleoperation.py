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
    """Base class for teleoperation with Reachy2 robot."""

    def __init__(self) -> None:
        """Initialize the teleoperation class.

        Loads the configuration from a YAML file and initializes the robot and controllers.
        """
        config = load_config("config.yaml")
        robot_ip = config.get("robot_ip", "localhost")
        mirror_mode = config.get("mirror_mode", False)
        self.control_mode = config.get("control_mode", "dual_arm")

        self.robot = Reachy2(robot_ip, mirror_mode)
        self.controllers: dict = {}
        self.controller_previous_pose = {LEFT_ARM: np.eye(4), RIGHT_ARM: np.eye(4)}
        self.robot_previous_pose = {LEFT_ARM: np.eye(4), RIGHT_ARM: np.eye(4)}

    @abstractmethod
    def init_teleoperation(self) -> None:
        pass

    def teleoperation(self) -> None:
        """Main loop for teleoperation.

        Initializes the teleoperation and enters a loop that calls the `step()` method at a fixed frequency.
        """
        self.init_teleoperation()
        frequency = 100
        while True:
            t = time.time()
            self.step()
            time.sleep(max(0, 1 / frequency - (time.time() - t)))

    @abstractmethod
    def step(self):
        pass


class TeleoperationRGBD(Teleoperation):
    """Teleoperation class with RGBD-type tracker."""

    def __init__(self) -> None:
        """Initialize the teleoperation class.

        Loads the configuration from a YAML file and initializes the robot and controllers.
        """
        super().__init__()
        self.camera = Orbbec()
        self.computer_vision = ComputerVision(self.camera)
        self.controllers = {
            LEFT_ARM: ArmRGBDController(self.computer_vision, LEFT_ARM),
            RIGHT_ARM: ArmRGBDController(self.computer_vision, RIGHT_ARM),
        }
        self.head_controller = HeadRGBDController(self.computer_vision)
        self.first_command_ok = False

    def init_teleoperation(self) -> None:
        """Initialize the teleoperation.

        Initializes the robot and controllers, and waits for the first command to be valid.
        """
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

    def is_first_command_ok(self, pose: np.ndarray, is_for_r_arm: bool) -> bool:
        """Check if the first command is in a specific area.

        Args:
            pose (np.ndarray): The pose of the controller.
            is_for_r_arm (bool): True if the pose is for the right arm, False if for the left arm.
        Returns:
            bool: True if the first command is valid, False otherwise.
        """
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

    def step(self) -> None:
        """Step called at each iteration of the teleoperation loop.

        Update the robot's state (arms, grippers, head), check for stop flag, and visualize the landmarks.
        """
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
    """Teleoperation class with Joystick-type Tracker."""

    def __init__(self, tracker_type: TrackerType) -> None:
        """Initialize the teleoperation class.
        Loads the configuration from a YAML file and initializes the robot and controllers.
        Args:
            tracker_type (TrackerType): The type of tracker to use (ARUCO or VIVE).
        """
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

        self._init_controllers()

        self.head_previous_pose = np.eye(3)
        self.antenna_previous_position = {RIGHT_ARM: 0, LEFT_ARM: 0}

        self.stop_count = 0

        self.joystick_mode = {LEFT_ARM: 0, RIGHT_ARM: 0}
        self.controller_previous_button = {RIGHT_ARM: [1, 1, 1], LEFT_ARM: [1, 1, 1]}

    def _init_controllers(self) -> None:
        """Initialize the controllers based on the control mode.

        If the control mode is DUAL_ARM, initialize both left and right arm controllers.
        Otherwise, initialize only the specified arm controller.
        """
        if self.control_mode == DUAL_ARM:
            self.controllers = {
                LEFT_ARM: JoystickController(self.tracker_type, LEFT_ARM, self.camera),
                RIGHT_ARM: JoystickController(self.tracker_type, RIGHT_ARM, self.camera),
            }
        else:
            self.controllers = {
                self.control_mode: JoystickController(self.tracker_type, self.control_mode, self.camera),
            }

    def init_teleoperation(self) -> None:
        """Initialize the teleoperation.

        Initializes the robot and controllers, and sets the initial poses for the robot and controllers.
        """
        self.robot.unfreeze(RIGHT_ARM)
        self.robot.init_robot()
        for controller in self.controllers.values():
            controller.init_controller()
            self.controller_previous_pose[controller.arm] = controller.get_controller_pose()

            self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)

    def controller_pose_to_robot_pose(self, controller_pose: np.ndarray, arm: str) -> np.ndarray:
        """Convert a controller pose to a robot pose.

        Args:
            controller_pose (np.ndarray): The pose of the controller.
            arm (str): The arm to which the controller is attached.
        Returns:
            np.ndarray: The corresponding robot pose.
        """
        robot_pose = self.robot_previous_pose[arm].copy()
        diff_position = controller_pose[:3, 3] - self.controller_previous_pose[arm][:3, 3]
        robot_pose[:3, 3] += diff_position
        diff_orientation = controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        robot_pose[:3, :3] = diff_orientation @ self.robot_previous_pose[arm][:3, :3]
        return robot_pose

    def controller_to_head_orientation(self, controller_pose: np.ndarray, arm: str) -> np.ndarray:
        """Convert a controller pose to a head orientation.
        Args:
            controller_pose (np.ndarray): The pose of the controller.
            arm (str): The arm to which the controller is attached.
        Returns:
            np.ndarray: The corresponding head orientation.
        """
        diff_orientation = controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        head_orientation = diff_orientation @ self.head_previous_pose
        return head_orientation

    def joystick_to_mobile_base(self, x_input: float, y_input: float) -> tuple[int, int]:
        """Convert joystick input to mobile base movement.

        Args:
            x_input (float): The x-axis input from the joystick.
            y_input (float): The y-axis input from the joystick.
        Returns:
            tuple[float, float]: The corresponding x and y movement commands for the mobile base.
        """
        command_max = 0.4
        center_joystick = 512
        x_goal, y_goal = 0.0, 0.0

        if not np.isclose(x_input, center_joystick, atol=70):
            y_goal = command_max / center_joystick * x_input - command_max
        if not np.isclose(y_input, center_joystick, atol=70):
            x_goal = command_max / center_joystick * y_input - command_max
        return int(x_goal), int(y_goal)

    def trigger_joint_to_gripper_joint(self, joint: int, arm: str) -> int:
        """Convert trigger joint position to gripper joint position.

        Args:
            joint (int): The position of the trigger joint.
            arm (str): The arm to which the trigger joint belongs.
        Returns:
            int: The corresponding gripper joint position.
        """
        if arm == "r_arm":
            min_joint, max_joint = -60, -15
        else:
            min_joint, max_joint = 60, 15
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper
        return int(gripper_opening)

    def joystick_to_antenna(self, input: int, arm: str) -> int:
        """Convert joystick input to antenna position, as a difference from the previous position.

        Args:
            input (int): The joystick input.
            arm (str): The arm to which the joystick belongs.
        Returns:
            float: The corresponding antenna position.
        """
        center_joystick = 512
        command_max = 2

        if not np.isclose(input, center_joystick, atol=70):
            delta = command_max / center_joystick * input - command_max
            antenna_position = self.antenna_previous_position[arm] + delta
        else:
            antenna_position = self.antenna_previous_position[arm]
        return int(antenna_position)

    def manage_mode(self) -> None:
        """Manage the control mode of the teleoperation.

        This method handles the stop button, mode toggle, and joystick toggle for each controller.
        It also updates the previous button states for each controller.
        """
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

    def _handle_stop_button(self, controller: JoystickController) -> None:
        """Handle the stop button for the teleoperation.
        If the stop button is pressed for a certain duration, toggle the stop state of the teleoperation.
        Args:
            controller (JoystickController): The controller to check for the stop button.
        """
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

    def _handle_mode_toggle(self, controller: JoystickController) -> None:
        """Handle the mode toggle button for the teleoperation.

        If the mode toggle button is pressed, switch between the two modes (0 and 1).
        Args:
            controller (JoystickController): The controller to check for the mode toggle button.
        """
        if controller.buttonA == 0 and controller.buttonA != self.controller_previous_button[controller.arm][1]:
            print("Change mode")
            self.mode = 0 if self.mode == 1 else 1

    def _handle_mode_set(self, controller: JoystickController) -> None:
        """Set the mode (switched by a press of the buttonA).
        Args:
            controller (JoystickController): The controller to check for the buttonA state.
        """
        if controller.buttonA == 0:
            self.mode = 2
        elif self.mode != 1:
            self.mode = 0

    def _handle_joystick_toggle(self, controller: JoystickController, single_arm: bool = False) -> None:
        """Handle the joystick toggle button for the teleoperation.

        If the joystick toggle button is pressed, switch between the two joystick modes (0 and 1).
        Args:
            controller (JoystickController): The controller to check for the joystick toggle button.
            single_arm (bool): True if the teleoperation is in single-arm mode, False otherwise.
              This is used to avoid changing the mode when the joystick button is pressed in single-arm mode.
        """
        if (
            controller.joystick_button == 0
            and controller.joystick_button != self.controller_previous_button[controller.arm][0]
        ):
            print("Change joystick mode")
            if single_arm:
                self.robot.unfreeze(controller.arm)
                self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)

            self.joystick_mode[controller.arm] = 0 if self.joystick_mode[controller.arm] == 1 else 1

    def _update_previous_buttons(self, controller: JoystickController) -> None:
        """Update the previous button states for the controller.

        Args:
            controller (JoystickController): The controller to update the previous button states for.
        """
        self.controller_previous_button[controller.arm] = [
            controller.joystick_button,
            controller.buttonA,
            controller.buttonB,
        ]

    def step(self) -> None:
        """Step called at each iteration of the teleoperation loop, updating the robot state."""
        # self.manage_mode()
        if not self.stop:
            self.update_robot_state()
            if tracker_type == TrackerType.ARUCO and isinstance(self.camera, RGBCamera):
                frame = self.get_video_streaming()
                cv2.imshow("Color Viewer", frame)
                cv2.waitKey(1)

    def update_robot_state(self) -> None:
        """Update the robot state based on the current control mode.

        This method updates the robot's arms, grippers, and head based on the current control mode.
        It also updates the previous poses for the robot and controllers.
        """
        if self.control_mode == DUAL_ARM:
            self.update_dual_arm_state()
        elif self.control_mode in [RIGHT_ARM, LEFT_ARM]:
            self.update_single_arm_state()
        else:
            raise ValueError(f"Invalid mode: {self.control_mode}")

    def update_dual_arm_state(self) -> None:
        """Update the robot state for dual-arm teleoperation."""
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

    def update_single_arm_state(self) -> None:
        """Update the robot state for single-arm teleoperation."""
        controller = self.controllers[self.control_mode]
        pose = controller.get_controller_pose()

        if self.mode == 0:
            robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm)
            self.robot.go_to_pose(robot_pose, controller.arm)
            self.robot_previous_pose[controller.arm] = robot_pose
            # x, y, theta = 0, 0, 0
            # joystick_x, joystick_y = self.joystick_to_mobile_base(controller.joystick_x, controller.joystick_y)
            # if self.joystick_mode[self.control_mode] == 0:
            #     x, y = joystick_x, joystick_y
            # else:
            #     x = joystick_x
            #     theta = joystick_y * 100
            # self.robot.move_mobile_base(x, y, theta)

        elif self.mode == 1:
            head_orientation = self.controller_to_head_orientation(pose, controller.arm)
            self.robot.move_head(head_orientation)
            self.head_previous_pose = head_orientation
            # antenna = self.joystick_to_antenna(controller.joystick_x, controller.arm)
            # self.robot.move_antenna(antenna)
            # self.antenna_previous_position[controller.arm] = antenna
        self.controller_previous_pose[controller.arm] = pose

    def get_video_streaming(self) -> Optional[np.ndarray]:
        """Get the video streaming from the Orbbec camera with the poses of the left and right ArUco cubes.

        Returns:
            np.ndarray: The camera frame with the poses of the left and right cubes.
        """
        if self.camera is not None and isinstance(self.camera, RGBCamera):
            if self.control_mode == DUAL_ARM:
                l_pose = self.controllers[LEFT_ARM].tracker.tracker_pose
                r_pose = self.controllers[RIGHT_ARM].tracker.tracker_pose
            elif self.control_mode == LEFT_ARM:
                l_pose = self.controllers[LEFT_ARM].tracker.tracker_pose
                r_pose = np.eye(4)
            elif self.control_mode == RIGHT_ARM:
                l_pose = np.eye(4)
                r_pose = self.controllers[RIGHT_ARM].tracker.tracker_pose
            frame = self.camera.get_frame_with_cube_pose([l_pose, r_pose])
            return frame
        else:
            raise AttributeError("Camera is not initialized or does not support 'get_frame_with_cube_pose'")


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
