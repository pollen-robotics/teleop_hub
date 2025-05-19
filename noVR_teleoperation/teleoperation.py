import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

# import cv2  # type: ignore
import numpy as np
import pygame

# from camera.camera import Camera  # type: ignore
# from camera.orbbec import Orbbec  # type: ignore
# from camera.rgb_camera import RGBCamera  # type: ignore
from controller.joystick_controller import JoystickController  # type: ignore

# from controller.so_arm_controller import SoArmController  # type: ignore
from pynput import keyboard

# from controller.rgbd_controller import ArmRGBDController, HeadRGBDController
from robots.reachy import Reachy2
from scipy.spatial.transform import Rotation as R  # type: ignore
from scipy.spatial.transform import Slerp

# from trackers.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from trackers.tracker import TrackerType  # type: ignore
from utils import axis_cleaner, load_config, parse_hat

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
        self.stop_flag = False

    @abstractmethod
    def init_teleoperation(self) -> None:
        pass

    def teleoperation(self) -> None:
        """Main loop for teleoperation.

        Initializes the teleoperation and enters a loop that calls the `step()` method at a fixed frequency.
        """
        self.init_teleoperation()
        frequency = 100
        while not self.stop_flag:
            t = time.time()
            self.step()
            time.sleep(max(0, 1 / frequency - (time.time() - t)))
        self.robot.stop()

    @abstractmethod
    def step(self):
        pass


class TeleoperationRGBD(Teleoperation):
    """Teleoperation class with RGBD-type controller."""

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
        self.user_offsets = np.zeros(3)

    def init_teleoperation(self) -> None:
        """Initialize the teleoperation.

        Initializes the robot and controllers, and waits for the first command to be valid,
        before starting the teleoperation.
        """
        self.robot.init_robot()
        robot_poses = [self.robot.fk(LEFT_ARM), self.robot.fk(RIGHT_ARM)]

        # check that the first commands are in a specific area before starting the teleoperation
        # and set the user offsets
        while not self.first_command_ok:
            is_pose_ok = [False, False]
            self.computer_vision.update_landmarks_coordinates(True)
            for i, controller in enumerate(self.controllers.values()):
                pose = controller.get_controller_pose()
                is_pose_ok[i] = controller.check_first_command(pose)
            if all(is_pose_ok):
                self.first_command_ok = True
                self._set_user_offsets(
                    [
                        self.controllers[LEFT_ARM].first_pose,
                        self.controllers[RIGHT_ARM].first_pose,
                    ],
                    robot_poses,
                )

    def _set_user_offsets(
        self, user_poses: list[np.ndarray], robot_poses: list[np.ndarray]
    ) -> None:
        """Set the user offsets based on the initial poses of the user and robot.

        The offsets are calculated as the difference between the robot and user positions in the x, y, and z axes,
        and they are stored in the `user_offsets` attribute.

        Args:
            user_poses (list[np.ndarray]): The initial poses of the user.
            robot_poses (list[np.ndarray]): The initial poses of the robot.
        """
        user_l_position = user_poses[0][:3, 3]
        user_r_position = user_poses[1][:3, 3]
        robot_l_position = robot_poses[0][:3, 3]
        robot_r_position = robot_poses[1][:3, 3]

        # calculate the offsets between the robot and user positions (as the mean of the two arms)
        x_offset = np.mean(
            [
                robot_l_position[0] - user_l_position[0],
                robot_r_position[0] - user_r_position[0],
            ]
        )
        y_offset = np.mean(
            [
                robot_l_position[1] - user_l_position[1],
                robot_r_position[1] - user_r_position[1],
            ]
        )
        z_offset = np.mean(
            [
                robot_l_position[2] - user_l_position[2],
                robot_r_position[2] - user_r_position[2],
            ]
        )

        self.user_offsets = np.array([x_offset, y_offset, z_offset])
        print(f"User offsets: {self.user_offsets}")

    def recalibrate_goal_position(self, pose: np.ndarray) -> np.ndarray:
        """Recalibrate the goal position based on the user offsets.

        Args:
            pose (np.ndarray): The pose of the controller.
        Returns:
            np.ndarray: The recalibrated pose.
        """
        pose[:3, 3] += self.user_offsets
        return pose

    def step(self) -> None:
        """Step called at each iteration of the teleoperation loop.

        Update the robot's state (arms, grippers, head), check for stop flag, and visualize the landmarks.
        """
        # Check if the stop flag is raised
        if self.head_controller.stop_flag:
            self.stop_flag = True
            return

        self.computer_vision.update_landmarks_coordinates(True)

        for controller in self.controllers.values():
            # Get the wrist command
            pose = controller.get_controller_pose()
            pose = self.recalibrate_goal_position(pose)
            self.robot.go_to_pose(pose, controller.arm)
            self.controller_previous_pose[controller.arm] = pose

            # Get the gripper command
            gripper_command = controller.get_gripper_command()
            if gripper_command is not None:
                self.robot.move_gripper(controller.arm, False, gripper_command)

        # Get the head command
        head_pose = self.head_controller.get_controller_pose()
        self.robot.move_head(head_pose)

        rpy = self.head_controller.former_rpy[-1]

        # Show the landmarks on the frame
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
                RIGHT_ARM: JoystickController(
                    self.tracker_type, RIGHT_ARM, self.camera
                ),
            }
        else:
            self.controllers = {
                self.control_mode: JoystickController(
                    self.tracker_type, self.control_mode, self.camera
                ),
            }

    def init_teleoperation(self) -> None:
        """Initialize the teleoperation.

        Initializes the robot and controllers, and sets the initial poses for the robot and controllers.
        """
        self.robot.unfreeze(RIGHT_ARM)
        self.robot.init_robot()
        for controller in self.controllers.values():
            controller.init_controller()
            self.controller_previous_pose[controller.arm] = (
                controller.get_controller_pose()
            )
            self.antenna_previous_position = {RIGHT_ARM: 0, LEFT_ARM: 0}
            self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)
            self.mode = 0
            self.head_previous_pose = np.eye(3)

    def controller_pose_to_robot_pose(
        self, controller_pose: np.ndarray, arm: str
    ) -> np.ndarray:
        """Convert a controller pose to a robot pose.

        Args:
            controller_pose (np.ndarray): The pose of the controller.
            arm (str): The arm to which the controller is attached.
        Returns:
            np.ndarray: The corresponding robot pose.
        """
        robot_pose = self.robot_previous_pose[arm].copy()
        diff_position = (
            controller_pose[:3, 3] - self.controller_previous_pose[arm][:3, 3]
        )
        robot_pose[:3, 3] += diff_position
        diff_orientation = (
            controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        )
        robot_pose[:3, :3] = diff_orientation @ self.robot_previous_pose[arm][:3, :3]
        return robot_pose

    def controller_to_head_orientation(
        self, controller_pose: np.ndarray, arm: str
    ) -> np.ndarray:
        """Convert a controller pose to a head orientation.
        Args:
            controller_pose (np.ndarray): The pose of the controller.
            arm (str): The arm to which the controller is attached.
        Returns:
            np.ndarray: The corresponding head orientation.
        """
        diff_orientation = (
            controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        )
        head_orientation = diff_orientation @ self.head_previous_pose
        return head_orientation

    def joystick_to_mobile_base(
        self, x_input: float, y_input: float
    ) -> tuple[int, int]:
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
        return x_goal, y_goal

    def trigger_joint_to_gripper_joint(self, joint: int, arm: str) -> int:
        """Convert trigger joint position to gripper joint position.

        Args:
            joint (int): The position of the trigger joint.
            arm (str): The arm to which the trigger joint belongs.
        Returns:
            int: The corresponding gripper joint position.
        """
        min_joint, max_joint = self.controllers[arm].gripper_joints_limit
        # if arm == "r_arm":
        #     min_joint, max_joint = -60, -20
        # else:
        #     min_joint, max_joint = 60, 20
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (
            max_gripper - min_gripper
        ) + min_gripper
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
        if (
            controller.buttonA == 0
            and controller.buttonA != self.controller_previous_button[controller.arm][1]
        ):
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

    def _handle_joystick_toggle(
        self, controller: JoystickController, single_arm: bool = False
    ) -> None:
        """Handle the joystick toggle button for the teleoperation.

        If the joystick toggle button is pressed, switch between the two joystick modes (0 and 1).
        Args:
            controller (JoystickController): The controller to check for the joystick toggle button.
            single_arm (bool): True if the teleoperation is in single-arm mode, False otherwise.
              This is used to avoid changing the mode when the joystick button is pressed in single-arm mode.
        """
        if (
            controller.joystick_button == 0
            and controller.joystick_button
            != self.controller_previous_button[controller.arm][0]
        ):
            print("Change joystick mode")
            if single_arm:
                self.robot.unfreeze(controller.arm)
                self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)

            self.joystick_mode[controller.arm] = (
                0 if self.joystick_mode[controller.arm] == 1 else 1
            )

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
        self.manage_mode()
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
                    head_orientation = self.controller_to_head_orientation(
                        pose, controller.arm
                    )
                    self.robot.move_head(head_orientation)
                    self.head_previous_pose = head_orientation
                else:
                    robot_pose = self.controller_pose_to_robot_pose(
                        pose, controller.arm
                    )
                    self.robot.go_to_pose(robot_pose, controller.arm)
                    self.robot_previous_pose[controller.arm] = robot_pose
                    gripper_joint = controller.get_gripper_joint()
                    gripper_joint = self.trigger_joint_to_gripper_joint(
                        gripper_joint, controller.arm
                    )
                    self.robot.move_gripper(controller.arm, True, gripper_joint)
                self.controller_previous_pose[controller.arm] = pose

            x, y, theta = 0, 0, 0
            if self.joystick_mode[RIGHT_ARM] == 0:
                _, r_y = self.joystick_to_mobile_base(
                    self.controllers[RIGHT_ARM].joystick_x,
                    self.controllers[RIGHT_ARM].joystick_y,
                )
                theta = r_y * 100

            else:
                antenna = self.joystick_to_antenna(
                    self.controllers[RIGHT_ARM].joystick_x, RIGHT_ARM
                )
                self.robot.move_antenna(antenna, RIGHT_ARM)
                self.antenna_previous_position[RIGHT_ARM] = antenna
            if self.joystick_mode[LEFT_ARM] == 0:
                x, y = self.joystick_to_mobile_base(
                    self.controllers[LEFT_ARM].joystick_x,
                    self.controllers[LEFT_ARM].joystick_y,
                )
            else:
                antenna = self.joystick_to_antenna(
                    self.controllers[LEFT_ARM].joystick_x, LEFT_ARM
                )
                self.robot.move_antenna(antenna, LEFT_ARM)
                self.antenna_previous_position[LEFT_ARM] = antenna
            self.robot.move_mobile_base(x, y, theta)

    def update_single_arm_state(self) -> None:
        """Update the robot state for single-arm teleoperation."""
        controller = self.controllers[self.control_mode]
        pose = controller.get_controller_pose()

        if self.mode == 0:
            robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm)
            self.robot.go_to_pose(robot_pose, controller.arm)
            self.robot_previous_pose[controller.arm] = robot_pose
            x, y, theta = 0, 0, 0
            joystick_x, joystick_y = self.joystick_to_mobile_base(
                controller.joystick_x, controller.joystick_y
            )
            if self.joystick_mode[self.control_mode] == 0:
                x, y = joystick_x, joystick_y
            else:
                x = joystick_x
                theta = joystick_y * 100
            self.robot.move_mobile_base(x, y, theta)
            gripper_joint = controller.get_gripper_joint()
            gripper_joint = self.trigger_joint_to_gripper_joint(
                gripper_joint, controller.arm
            )
            self.robot.move_gripper(controller.arm, True, gripper_joint)

        elif self.mode == 1:
            head_orientation = self.controller_to_head_orientation(pose, controller.arm)
            self.robot.move_head(head_orientation)
            self.head_previous_pose = head_orientation
            antenna = self.joystick_to_antenna(controller.joystick_x, controller.arm)
            self.robot.move_antenna(antenna)
            self.antenna_previous_position[controller.arm] = antenna
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
            raise AttributeError(
                "Camera is not initialized or does not support 'get_frame_with_cube_pose'"
            )


class SoArmTeleoperation(Teleoperation):
    def __init__(self, port):
        super().__init__()

        self.so_arm_controller = SoArmController(port)

        time.sleep(1)
        print("ok")
        self.robot_part = "r_arm"
        self.init = False
        self.pause = False
        self.mouse = False
        self.top_grasp = {"r_arm": False, "l_arm": False}
        self.init_tg = False
        self.position_coeff = 1
        self.orientation_coeff = 1
        self.mobile_base = False
        self.mirror = False

        self.so_previous_pose = {
            "r_arm": self.so_arm_controller.so_arm_fk(
                self.so_arm_controller.so_previous_joints["r_arm"]
            ),
            "l_arm": self.so_arm_controller.so_arm_fk(
                self.so_arm_controller.so_previous_joints["l_arm"]
            ),
            "head": self.so_arm_controller.so_arm_fk(
                self.so_arm_controller.so_previous_joints["head"]
            ),
            "mobile_base": self.so_arm_controller.so_arm_fk(
                self.so_arm_controller.so_previous_joints["mobile_base"]
            ),
        }

    def init_teleoperation(self):
        print("init teleoperation")
        self.robot.init_robot()

        self.so_arm_controller.init_controller(self.robot_part)

        self.robot_previous_pose = {
            "r_arm": self.robot.fk("r_arm"),
            "l_arm": self.robot.fk("l_arm"),
        }
        self.real_robot_previous_pose = {
            "r_arm": self.robot.fk("r_arm"),
            "l_arm": self.robot.fk("l_arm"),
        }
        self.robot_head_joints = [0, 0, 0]

        keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        keyboard_thread.start()

    def step(self):
        part = self.robot_part

        if part == "l_arm" or part == "r_arm":
            top_grasp = self.top_grasp[part]

        if self.pause:
            self.robot.move_mobile_base(x=0, y=0, theta=0)
        elif self.mouse:
            self.so_arm_controller.update_joints(part)
            self.so_previous_pose[part] = self.so_arm_controller.get_controller_pose()
        elif self.init:
            print("switching arm")
            self.so_arm_controller.init_controller(part)
            print("done")
            self.init = False
        elif self.init_tg:
            print("init top grasp")
            self.init_top_grasp()
            self.init_tg = False
        elif part == "head":
            so_joints = self.so_arm_controller.get_head_joints()
            head_joints = self.so_joints_to_head_joints(so_joints)
            self.robot.move_head(head_joints)
            trigger_joint = self.so_arm_controller.get_gripper_joint()
            antenna_joint = self.trigger_joint_to_gripper_joint(trigger_joint, part)
            self.robot.move_antenna(antenna_joint)
            self.so_arm_controller.update_joints(part)
        elif part == "mobile_base":
            so_pose = self.so_arm_controller.get_controller_pose()
            x, y, theta = self.so_pose_to_mobile_base_speed(so_pose)
            self.robot.move_mobile_base(x, y, theta)
        else:
            pose = self.so_arm_controller.get_controller_pose()
            self.so_arm_controller.update_joints(part)
            robot_pose = self.find_robot_pose(pose, part, top_grasp)
            self.robot_previous_pose[part] = robot_pose
            self.robot.go_to_pose(robot_pose, part)
            controller_joint = self.so_arm_controller.get_gripper_joint()
            trigger_joint = self.trigger_joint_to_gripper_joint(controller_joint, part)
            self.robot.move_gripper(part, True, trigger_joint)

    def init_top_grasp(self):
        self.so_arm_controller.lock_arm()
        print(self.top_grasp[self.robot_part])
        start_orientation = R.from_matrix(
            self.robot_previous_pose[self.robot_part][:3, :3]
        )
        so_pose = self.so_arm_controller.get_controller_pose()

        pose = self.find_robot_pose(
            so_pose, self.robot_part, self.top_grasp[self.robot_part]
        )
        self.robot_previous_pose[self.robot_part] = pose
        end_orientation = R.from_matrix(pose[:3, :3])

        frequency = 100  # Nombre de mises à jour par seconde
        duration = 2.0  # Durée totale de l'interpolation en secondes
        steps = int(duration * frequency)

        slerp = Slerp([0, 1], R.concatenate([start_orientation, end_orientation]))

        for i in range(steps):
            alpha = i / steps  # Facteur d'interpolation
            interpolated_rotation = slerp([alpha])  # Interpolation à l'instant alpha
            pose[:3, :3] = interpolated_rotation.as_matrix()[0]  # Convertir en matrice*
            self.robot.go_to_pose(pose, self.robot_part)
            time.sleep(1 / frequency)
        self.robot_previous_pose[self.robot_part] = pose
        self.so_arm_controller.unlock_arm()

    def on_press(self, key):
        print(f"Key {key} pressed")
        try:
            if key.char == "l":
                print("press l")
                if (
                    not self.init
                    and not self.pause
                    and self.robot_part != "l_arm"
                    and not self.mouse
                    and not self.init_tg
                ):
                    self.init = True
                    self.robot_part = "l_arm"

            elif key.char == "r":
                if (
                    not self.init
                    and not self.pause
                    and self.robot_part != "r_arm"
                    and not self.mouse
                    and not self.init_tg
                ):
                    self.init = True
                    self.robot_part = "r_arm"

            elif key.char == "p":
                if not self.init and not self.init_tg:
                    self.pause = not self.pause
                    if self.pause:
                        self.so_arm_controller.lock_arm()
                    else:
                        self.so_arm_controller.unlock_arm()

            elif key.char == "o":
                if not self.init and not self.pause:
                    self.mouse = not self.mouse

            elif key.char == "t":
                print(self.init, self.pause, self.init_tg, self.mouse)
                if (
                    not self.init
                    and not self.pause
                    and not self.init_tg
                    and not self.mouse
                ):
                    self.init_tg = True
                    self.top_grasp[self.robot_part] = not self.top_grasp[
                        self.robot_part
                    ]

            elif key.char == "h":
                if (
                    not self.init
                    and not self.pause
                    and not self.init_tg
                    and not self.mouse
                ):
                    self.init = True
                    self.robot_part = "head"

            elif key.char == "i":
                self.position_coeff += 0.1
                print(f"Position coeff: {self.position_coeff}")

            elif key.char == "k":
                self.position_coeff -= 0.1
                print(f"Position coeff: {self.position_coeff}")

            elif key.char == "u":
                self.orientation_coeff += 0.1
                print(f"Orientation coeff: {self.orientation_coeff}")

            elif key.char == "j":
                self.orientation_coeff -= 0.1
                print(f"Orientation coeff: {self.orientation_coeff}")

            elif key.char == "m":
                if (
                    not self.init
                    and not self.pause
                    and not self.init_tg
                    and not self.mouse
                ):
                    self.init = True
                    self.robot_part = "mobile_base"

            elif key.char == "n":
                self.mirror = not self.mirror
                print(f"Mirror: {self.mirror}")

        except AttributeError:
            print(f"Key {key} pressed")

    def listen_keyboard(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def find_robot_pose(self, so_pose, arm, top_grasp):
        if top_grasp:
            orientation = so_pose[:3, :3]
            if arm == "r_arm":
                rot1 = R.from_euler(
                    "xyz", [0, 0, -np.pi / 4], degrees=False
                ).as_matrix()
            else:
                rot1 = R.from_euler("xyz", [0, 0, np.pi / 4], degrees=False).as_matrix()
            rot2 = R.from_euler("xyz", [0, np.pi / 2, 0], degrees=False).as_matrix()
            orientation = orientation @ rot2 @ rot1
            so_pose[:3, :3] = orientation

        robot_pose = self.robot_previous_pose[arm].copy()

        diff_position = so_pose[:3, 3] - self.so_previous_pose[arm][:3, 3]
        diff_position *= self.position_coeff
        if self.mirror:
            diff_position[1] = -diff_position[1]
        robot_pose[:3, 3] += diff_position

        diff_orientation = so_pose[:3, :3] @ self.so_previous_pose[arm][:3, :3].T
        diff_orientation = R.from_matrix(diff_orientation).as_euler(
            "xyz", degrees=False
        )
        diff_orientation *= self.orientation_coeff
        if self.mirror:
            diff_orientation[0] = -diff_orientation[0]
            diff_orientation[2] = -diff_orientation[2]
        diff_orientation = R.from_euler(
            "xyz", diff_orientation, degrees=False
        ).as_matrix()
        robot_pose[:3, :3] = diff_orientation @ self.robot_previous_pose[arm][:3, :3]

        self.so_previous_pose[arm] = so_pose

        return robot_pose

    def trigger_joint_to_gripper_joint(self, joint, arm):
        joint = np.rad2deg(joint)
        min_joint, max_joint = -60, -15
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (
            max_gripper - min_gripper
        ) + min_gripper

        return int(gripper_opening)

    def trigger_joint_to_antenna_joint(self, joint):
        min_joint, max_joint = -60, 0
        min_antenna, max_antenna = 30, -160
        if joint < -60:
            joint = -60
        if joint > -0:
            joint = -0
        antenna_opening = ((joint - min_joint) / (max_joint - min_joint)) * (
            max_antenna - min_antenna
        ) + min_antenna

        return int(antenna_opening)

    def so_joints_to_head_joints(self, joints):
        joints = np.rad2deg(joints)
        robot_roll_range = [-40, 40]
        robot_pitch_range = [-40, 40]
        so_roll_range = [-100, 0]
        so_pitch_range = [-40, 40]

        head_roll = np.interp(joints[0], so_roll_range, robot_roll_range)
        head_pitch = np.interp(joints[1], so_pitch_range, robot_pitch_range)
        head_yaw = joints[2] + 65

        orientation_matrix = R.from_euler(
            "xyz", [head_roll, head_pitch, head_yaw], degrees=True
        ).as_matrix()

        return orientation_matrix

    def so_pose_to_mobile_base_speed(self, so_pose):
        diff_position = so_pose[:3, 3] - self.so_previous_pose["mobile_base"][:3, 3]

        x = diff_position[0] * 5
        y = diff_position[1] * 5
        diff_orientation = (
            so_pose[:3, :3] @ self.so_previous_pose["mobile_base"][:3, :3].T
        )
        diff_orientation = R.from_matrix(diff_orientation).as_euler(
            "xyz", degrees=False
        )

        theta = diff_orientation[2]
        theta = np.rad2deg(theta)

        return x, y, theta


class DualShockTeleoperation(Teleoperation):
    """Teleoperation using a DualShock controller, driving Reachy2 with the generic robot interface."""

    def __init__(self, angle_step, x_y_joystick_ratio, z_increment) -> None:
        super().__init__()
        # Initialize pygame joystick
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No DualShock controller found")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.ANGLE_STEP = angle_step
        self.X_Y_JOYSTICK_RATIO = x_y_joystick_ratio
        self.Z_INCREMENT = z_increment
        self.continous_press = [False] * 12

        # Robot previous poses for incremental updates
        self.robot.init_robot()
        self.robot_previous_pose = {
            LEFT_ARM: self.robot.fk(LEFT_ARM),
            RIGHT_ARM: self.robot.fk(RIGHT_ARM),
        }

        # Deltas from controller inputs
        self.controller_delta = {
            LEFT_ARM: {k: 0.0 for k in ("dx", "dy", "dz", "roll", "pitch", "yaw")},
            RIGHT_ARM: {k: 0.0 for k in ("dx", "dy", "dz", "roll", "pitch", "yaw")},
        }

        # Gripper states: 0=closed, 1=open
        self.gripper_state = {LEFT_ARM: 0, RIGHT_ARM: 0}
        self._stop_flag = False

    def init_teleoperation(self) -> None:
        """Initialize robot arms before teleoperation loop starts."""
        self.robot.init_robot()
        for arm in (LEFT_ARM, RIGHT_ARM):
            self.robot.unfreeze(arm)
            self.robot_previous_pose[arm] = self.robot.fk(arm)

    def step(self) -> None:
        """Read DualShock inputs, update robot target poses, grippers, and send commands."""
        pygame.event.pump()

        # reset deltas
        for arm in (LEFT_ARM, RIGHT_ARM):
            for k in self.controller_delta[arm].keys():
                self.controller_delta[arm][k] = 0.0

        # --- RIGHT ARM control ---
        # Right stick -> X/Y
        rx = axis_cleaner(self.joystick.get_axis(3))
        ry = axis_cleaner(self.joystick.get_axis(4))
        self.controller_delta[RIGHT_ARM]["dx"] = -ry * self.X_Y_JOYSTICK_RATIO
        self.controller_delta[RIGHT_ARM]["dy"] = -rx * self.X_Y_JOYSTICK_RATIO
        # R1/R2 -> Z
        if self.joystick.get_button(5):
            self.controller_delta[RIGHT_ARM]["dz"] = self.Z_INCREMENT
        elif self.joystick.get_button(7):
            self.controller_delta[RIGHT_ARM]["dz"] = -self.Z_INCREMENT

        # Right RPY
        hat = self.joystick.get_hat(0)
        if hat != (0, 0):
            dx_hat, dy_hat = parse_hat(hat)
            if dx_hat == 0 and dy_hat == 1:
                self.controller_delta[RIGHT_ARM]["pitch"] = self.ANGLE_STEP
            elif dx_hat == 0 and dy_hat == -1:
                self.controller_delta[RIGHT_ARM]["pitch"] = -self.ANGLE_STEP
            elif dx_hat == 1 and dy_hat == 0:
                self.controller_delta[RIGHT_ARM]["yaw"] = -self.ANGLE_STEP
            elif dx_hat == -1 and dy_hat == 0:
                self.controller_delta[RIGHT_ARM]["yaw"] = self.ANGLE_STEP
            elif dx_hat == -1 and dy_hat == -1:
                self.controller_delta[RIGHT_ARM]["roll"] = -self.ANGLE_STEP
            elif dx_hat == 1 and dy_hat == -1:
                self.controller_delta[RIGHT_ARM]["roll"] = self.ANGLE_STEP

        # SHARE (button 8) -> toggle right gripper
        if self.joystick.get_button(8):
            if not self.continous_press[8]:
                self.continous_press[8] = True
                self.gripper_state[RIGHT_ARM] ^= 1
                self.robot.move_gripper(
                    RIGHT_ARM, True, 130 * self.gripper_state[RIGHT_ARM]
                )
                time.sleep(0.2)
            else:
                pass
        else:
            self.continous_press[8] = False

        # --- LEFT ARM control ---
        # Left stick -> X/Y
        lx = axis_cleaner(self.joystick.get_axis(0))
        ly = axis_cleaner(self.joystick.get_axis(1))
        self.controller_delta[LEFT_ARM]["dx"] = -ly * self.X_Y_JOYSTICK_RATIO
        self.controller_delta[LEFT_ARM]["dy"] = -lx * self.X_Y_JOYSTICK_RATIO
        # L1/L2 -> Z
        if self.joystick.get_button(4):
            self.controller_delta[LEFT_ARM]["dz"] = self.Z_INCREMENT
        elif self.joystick.get_button(6):
            self.controller_delta[LEFT_ARM]["dz"] = -self.Z_INCREMENT

        # Left RPY
        if self.joystick.get_button(0):
            if self.joystick.get_button(3):  # combo
                self.controller_delta[LEFT_ARM]["roll"] = -self.ANGLE_STEP
            elif self.joystick.get_button(1):  # combo
                self.controller_delta[LEFT_ARM]["roll"] = self.ANGLE_STEP
            else:
                self.controller_delta[LEFT_ARM]["pitch"] = -self.ANGLE_STEP
        else:
            if self.joystick.get_button(3):
                self.controller_delta[LEFT_ARM]["yaw"] = self.ANGLE_STEP
            if self.joystick.get_button(1):
                self.controller_delta[LEFT_ARM]["yaw"] = -self.ANGLE_STEP
        if self.joystick.get_button(2):
            self.controller_delta[LEFT_ARM]["pitch"] = self.ANGLE_STEP
        # OPTIONS (button 9) -> toggle left gripper
        if self.joystick.get_button(9):
            if not self.continous_press[9]:
                self.continous_press[9] = True
                self.gripper_state[LEFT_ARM] ^= 1
                self.robot.move_gripper(
                    LEFT_ARM, True, 130 * self.gripper_state[LEFT_ARM]
                )
                time.sleep(0.2)
            else:
                pass
        else:
            self.continous_press[9] = False

        # PS button -> stop
        if self.joystick.get_button(10):
            self._stop_flag = True

        # Apply incremental movement and orientation to both arms
        for arm, delta in self.controller_delta.items():
            prev = self.robot_previous_pose[arm]
            new_pose = prev.copy()
            # translation
            new_pose[:3, 3] += np.array([delta["dx"], delta["dy"], delta["dz"]])
            # orientation
            rot_delta = R.from_euler(
                "xyz", [delta["roll"], delta["pitch"], delta["yaw"]], degrees=True
            ).as_matrix()
            new_pose[:3, :3] = rot_delta.dot(prev[:3, :3])
            # send command
            self.robot.go_to_pose(new_pose, arm)
            # update for next iteration
            self.robot_previous_pose[arm] = new_pose

        if self._stop_flag:
            self.stop_flag = True


if __name__ == "__main__":
    config = load_config("config.yaml")
    tracker_type_str = config.get("tracker_type", TrackerType.ARUCO).upper()
    tracker_type = getattr(TrackerType, tracker_type_str, None)

    teleoperation: Teleoperation

    if tracker_type == TrackerType.ARUCO or tracker_type == TrackerType.VIVE:
        teleoperation = JoystickTeleoperation(tracker_type)
    elif tracker_type == TrackerType.RGBD:
        teleoperation = TeleoperationRGBD()
    elif tracker_type == TrackerType.SO_ARM:
        teleoperation = SoArmTeleoperation(config["port"])
    elif tracker_type == TrackerType.DUALSHOCK:
        teleoperation = DualShockTeleoperation(
            config["angle_step"], config["x_y_joystick_ratio"], config["z_increment"]
        )
    else:
        raise ValueError(f"Invalid tracker type: {tracker_type}")

    try:
        teleoperation.teleoperation()

    except KeyboardInterrupt:
        for controller in teleoperation.controllers.values():
            controller.stop()
        teleoperation.robot.stop()
