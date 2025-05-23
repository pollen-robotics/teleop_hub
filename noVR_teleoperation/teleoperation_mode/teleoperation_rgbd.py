import numpy as np

from camera.orbbec import Orbbec  # type: ignore
from controller.rgbd_controller import (ArmRGBDController,  # type: ignore
                                        HeadRGBDController)
from teleoperation_mode.teleoperation import Teleoperation  # type: ignore
from teleoperation_mode.teleoperation import LEFT_ARM, RIGHT_ARM  # type: ignore
from trackers.rgbd_tracker.computer_vision import ComputerVision  # type: ignore


class TeleoperationRGBD(Teleoperation):
    """Teleoperation class with RGBD-type tracker."""

    def __init__(self) -> None:
        """Initialize the teleoperation class.

        Loads the configuration from a YAML file and initializes the robot and controllers.
        """
        super().__init__()

        self.camera = Orbbec()
        self.cv2 = self.camera.cv2
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
                    [self.controllers[LEFT_ARM].first_pose, self.controllers[RIGHT_ARM].first_pose], robot_poses
                )

    def _set_user_offsets(self, user_poses: list[np.ndarray], robot_poses: list[np.ndarray]) -> None:
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
        x_offset = np.mean([robot_l_position[0] - user_l_position[0], robot_r_position[0] - user_r_position[0]])
        y_offset = np.mean([robot_l_position[1] - user_l_position[1], robot_r_position[1] - user_r_position[1]])
        z_offset = np.mean([robot_l_position[2] - user_l_position[2], robot_r_position[2] - user_r_position[2]])

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

    def step(self):
        """Step called at each iteration of the teleoperation loop.

        Update the robot's state (arms, grippers, head), check for stop flag, and visualize the landmarks.
        """
        # Check if the stop flag is raised
        if self.head_controller.stop_flag:
            print("Stop flag raised, stopping teleoperation.")
            self.robot.stop()
            self.stop_flag = True

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
        self.camera.cv2.imshow("Color Viewer", frame)
        self.camera.cv2.waitKey(1)
