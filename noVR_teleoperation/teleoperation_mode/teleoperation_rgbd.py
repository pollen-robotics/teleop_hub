import time

import numpy as np

from camera.orbbec import Orbbec  # type: ignore
from controller.rgbd_controller import (ArmRGBDController,  # type: ignore
                                        HeadRGBDController)
from teleoperation_mode.teleoperation import Teleoperation  # type: ignore
from teleoperation_mode.teleoperation import LEFT_ARM, RIGHT_ARM
from trackers.rgbd_tracker.computer_vision import \
    ComputerVision  # type: ignore


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

        # check that the command is in a specific area before starting the teleoperation
        while not self.first_command_ok:
            self.computer_vision.update_landmarks_coordinates(True)
            l_pose = self.controllers[LEFT_ARM].get_controller_pose()
            r_pose = self.controllers[RIGHT_ARM].get_controller_pose()
            if self.is_first_command_ok(l_pose, False) and self.is_first_command_ok(
                r_pose, True
            ):
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
        self.camera.cv2.imshow("Color Viewer", frame)
        self.camera.cv2.waitKey(1)
