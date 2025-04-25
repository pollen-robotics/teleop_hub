import time

import cv2  # type: ignore
import numpy as np
from camera.camera import Camera  # type: ignore
from camera.orbbec import Orbbec  # type: ignore
from controller.controller import Controller
from controller.head_controller import HeadController  # type: ignore
from controller.rgbd_tracker.computer_vision import ComputerVision  # type: ignore
from controller.tracker import TrackerType  # type: ignore
from robots.reachy import Reachy2

DUAL_ARM = "dual_arm"
LEFT_ARM = "l_arm"
RIGHT_ARM = "r_arm"
MODE = DUAL_ARM


class Teleoperation:
    def __init__(self, tracker_type: TrackerType, marker_size=[0.06, 0.03], mirror_mode=False):
        self.tracker_type = tracker_type
        self.mirror_mode = mirror_mode

        self.stop = True
        self.mode = 0

        self.controllers = self.initialize_controllers(tracker_type, marker_size)
        if self.tracker_type == TrackerType.RGBD:
            self.head_controller = HeadController(self.computer_vision, self.mirror_mode)

        self.robot: Reachy2 = Reachy2(mirror_mode)

        self.controller_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}
        self.robot_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}
        self.head_previous_pose = np.eye(3)
        self.antenna_previous_position = {RIGHT_ARM: 0, LEFT_ARM: 0}

        self.stop_count = 0

        self.joystick_mode = {LEFT_ARM: 0, RIGHT_ARM: 0}
        self.controller_previous_button = {RIGHT_ARM: [1, 1, 1], LEFT_ARM: [1, 1, 1]}

    def initialize_controllers(self, tracker_type, marker_size):
        if tracker_type == TrackerType.ARUCO:
            return self.initialize_aruco_controllers(marker_size)
        elif tracker_type == TrackerType.VIVE:
            return self.initialize_vive_controllers()
        elif tracker_type == TrackerType.RGBD:
            return self.initialize_rgbd_controllers()
        else:
            raise ValueError(f"Invalid controller type: {tracker_type}")

    def initialize_aruco_controllers(self, marker_size):
        self.camera = Camera()
        self.marker_size = marker_size
        self.computer_vision = None
        self.with_buttons = True
        self.with_joint_command = True
        self.differential_mode = True
        return self.create_controllers()

    def initialize_vive_controllers(self):
        self.camera = None
        self.marker_size = np.zeros(2)
        self.computer_vision = None
        self.with_buttons = True
        self.with_joint_command = True
        self.differential_mode = True
        return self.create_controllers()

    def initialize_rgbd_controllers(self):
        self.camera = Orbbec()
        self.computer_vision = ComputerVision(self.camera)
        self.marker_size = np.zeros(2)
        self.with_buttons = False
        self.with_joint_command = False
        self.differential_mode = False
        return self.create_controllers()

    def create_controllers(self):
        if MODE == DUAL_ARM:
            return {
                LEFT_ARM: Controller(
                    self.tracker_type,
                    LEFT_ARM,
                    self.camera,
                    self.marker_size[0],
                    self.mirror_mode,
                    self.computer_vision,
                ),
                RIGHT_ARM: Controller(
                    self.tracker_type,
                    RIGHT_ARM,
                    self.camera,
                    self.marker_size[1],
                    self.mirror_mode,
                    self.computer_vision,
                ),
            }

        elif MODE == LEFT_ARM:
            return {
                LEFT_ARM: Controller(
                    self.tracker_type,
                    LEFT_ARM,
                    self.camera,
                    self.marker_size[0],
                    self.mirror_mode,
                    self.computer_vision,
                )
            }
        elif MODE == RIGHT_ARM:
            return {
                RIGHT_ARM: Controller(
                    self.tracker_type,
                    RIGHT_ARM,
                    self.camera,
                    self.marker_size[1],
                    self.mirror_mode,
                    self.computer_vision,
                )
            }
        else:
            raise ValueError(f"Invalid mode: {MODE}")

    def init_teleoperation(self):
        self.robot.unfreeze("r_arm")
        self.robot.init_robot()
        self.head_previous_pose = np.eye(3)
        self.antenna_previous_position = {RIGHT_ARM: 0, LEFT_ARM: 0}

        for controller in self.controllers.values():
            controller.init_controller()
            self.controller_previous_pose[controller.arm] = controller.get_controller_pose()

            self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)
        print("Teleoperation initialized.")

    def controller_pose_to_robot_pose(self, controller_pose, arm):
        if self.differential_mode:
            return self.compute_differential_arm_pose(controller_pose, arm)
        else:
            return controller_pose.copy()

    def compute_differential_arm_pose(self, controller_pose, arm):
        robot_pose = self.robot_previous_pose[arm].copy()
        diff_position = controller_pose[:3, 3] - self.controller_previous_pose[arm][:3, 3]
        robot_pose[:3, 3] += diff_position
        diff_orientation = controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        robot_pose[:3, :3] = diff_orientation @ self.robot_previous_pose[arm][:3, :3]
        return robot_pose

    def controller_to_head_orientation(self, controller_pose, arm):
        if self.differential_mode:
            return self.compute_differential_head_pose(controller_pose, arm)
        else:
            return controller_pose.copy()

    def compute_differential_head_pose(self, controller_pose, arm):
        print("differential head pose")
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
        if self.with_buttons:
            if MODE == DUAL_ARM:
                self._handle_stop_button(self.controllers[LEFT_ARM])
                self._handle_mode_toggle(self.controllers[RIGHT_ARM])
                self._handle_mode_set(self.controllers[LEFT_ARM])
                self._handle_joystick_toggle(self.controllers[RIGHT_ARM])
                self._handle_joystick_toggle(self.controllers[LEFT_ARM])

                for controller in self.controllers.values():
                    self._update_previous_buttons(controller)

            elif MODE in [LEFT_ARM, RIGHT_ARM]:
                controller = self.controllers[MODE]
                self._handle_stop_button(controller)
                self._handle_mode_toggle(controller)
                self._handle_joystick_toggle(controller, single_arm=True)
                self._update_previous_buttons(controller)
        else:
            

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
        if controller.joystick_button == 0 and controller.joystick_button != self.controller_previous_button[controller.arm][0]:
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

    def teleoperation(self):
        self.init_teleoperation()
        frequency = 100
        while True:
            t = time.time()
            if self.with_buttons:
                self.manage_mode()
            if self.computer_vision is not None:
                self.computer_vision.update_landmarks_coordinates(True)
                self.stop = False  # à changer avec la condition d'espace de travail
            if not self.stop:
                self.update_robot_state()
            if self.tracker_type == TrackerType.RGBD:
                color_frame = self.computer_vision.camera.color_frame[0]
                frame = self.computer_vision.visualization_landmarks(
                    color_frame,
                    self.controller_previous_pose[LEFT_ARM],
                    self.controller_previous_pose[RIGHT_ARM],
                    0,
                    0,
                    0,
                    text_on=True,
                )
                cv2.imshow("Color Viewer", frame)
                cv2.waitKey(1)
            time.sleep(max(0, 1 / frequency - (time.time() - t)))

    def update_robot_state(self):
        if MODE == DUAL_ARM:
            self.update_dual_arm_state()
        elif MODE in [RIGHT_ARM, LEFT_ARM]:
            self.update_single_arm_state()
        else:
            raise ValueError(f"Invalid mode: {MODE}")

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
                    gripper_command = controller.get_gripper_command()
                    if gripper_command is not None:
                        self.robot.move_gripper(controller.arm, self.with_joint_command, gripper_command)
                    if self.tracker_type == TrackerType.RGBD:
                        head_pose = self.head_controller.get_controller_pose()
                        self.robot.move_head(head_pose)
                        self.controller_to_head_orientation(head_pose, "head")
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
        controller = self.controllers[MODE]
        pose = controller.get_controller_pose()
        if self.mode == 0:
            robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm)
            self.robot.go_to_pose(robot_pose, controller.arm)
            self.robot_previous_pose[controller.arm] = robot_pose
            # x, y, theta = 0, 0, 0
            # joystick_x, joystick_y = self.joystick_to_mobile_base(
            #     controller.joystick_x, controller.joystick_y
            # )
            # if self.joystick_mode[MODE] == 0:
            #     x, y = joystick_x, joystick_y
            # else:
            #     x = joystick_x
            #     theta = joystick_y * 100
            # self.robot.move_mobile_base(x, y, theta)
        elif self.mode == 1:
            head_orientation = self.controller_to_head_orientation(pose, controller.arm)
            self.robot.move_head(head_orientation)
            self.head_previous_pose = head_orientation
            antenna = self.joystick_to_antenna(controller.joystick_x, controller.arm)
            self.robot.move_antenna(antenna)
            self.antenna_previous_position[controller.arm] = antenna
        self.controller_previous_pose[controller.arm] = pose

    # def teleoperation(self):
    #     self.init_teleoperation()
    #     frequency = 100
    #     while True:
    #         t = time.time()
    #         if self.with_buttons:
    #             self.manage_mode()
    #         if self.computer_vision is not None:
    #             self.computer_vision.update_landmarks_coordinates(True)
    #             self.stop = False
    #         if not self.stop:
    #             if MODE == DUAL_ARM:
    #                 print(f"mode: {self.mode}")
    #                 if self.mode == 2:
    #                     for controller in self.controllers.values():
    #                         pose = controller.get_controller_pose()
    #                         self.controller_previous_pose[controller.arm] = pose
    #                         self.robot.unfreeze(controller.arm)
    #                         self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)
    #                 else:
    #                     for controller in self.controllers.values():
    #                         pose = controller.get_controller_pose()
    #                         print(f"{controller.arm} pose: {pose}")

    #                         if controller.arm == RIGHT_ARM and self.mode == 1:
    #                             head_orientation = self.controller_to_head_orientation(pose, controller.arm)
    #                             self.robot.move_head(head_orientation)
    #                             self.head_previous_pose = head_orientation
    #                         else:
    #                             robot_pose = self.controller_pose_to_robot_pose(
    #                                 pose, controller.arm, self.differential_mode
    #                             )
    #                             self.robot.go_to_pose(robot_pose, controller.arm)
    #                             self.robot_previous_pose[controller.arm] = robot_pose
    #                             gripper_command = controller.get_gripper_command()
    #                             if gripper_command is not None:
    #                                 self.robot.move_gripper(controller.arm, self.with_joint_command, gripper_command)
    #                                 print(f"Gripper command: {gripper_command}")

    #                         self.controller_previous_pose[controller.arm] = pose

    #                     # x, y, theta = 0, 0, 0
    #                     # if self.joystick_mode[RIGHT_ARM] == 0:
    #                     #     _, r_y = self.joystick_to_mobile_base(
    #                     #         self.controllers[RIGHT_ARM].joystick_x,
    #                     #         self.controllers[RIGHT_ARM].joystick_y,
    #                     #     )
    #                     #     theta = r_y * 100

    #                     # else:
    #                     #     antenna = self.joystick_to_antenna(self.controllers[RIGHT_ARM].joystick_x, RIGHT_ARM)
    #                     #     self.robot.move_antenna(antenna, RIGHT_ARM)
    #                     #     self.antenna_previous_position[RIGHT_ARM] = antenna
    #                     # if self.joystick_mode[LEFT_ARM] == 0:
    #                     #     x, y = self.joystick_to_mobile_base(
    #                     #         self.controllers[LEFT_ARM].joystick_x,
    #                     #         self.controllers[LEFT_ARM].joystick_y,
    #                     #     )
    #                     # else:
    #                     #     antenna = self.joystick_to_antenna(self.controllers[LEFT_ARM].joystick_x, LEFT_ARM)
    #                     #     self.robot.move_antenna(antenna, LEFT_ARM)
    #                     #     self.antenna_previous_position[LEFT_ARM] = antenna
    #                     # self.robot.move_mobile_base(x, y, theta)

    #             elif MODE == RIGHT_ARM or MODE == LEFT_ARM:
    #                 controller = self.controllers[MODE]
    #                 pose = controller.get_controller_pose()

    #                 if self.mode == 0:
    #                     robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm, self.differential_mode)
    #                     self.robot.go_to_pose(robot_pose, controller.arm)
    #                     self.robot_previous_pose[controller.arm] = robot_pose
    #                     # x, y, theta = 0, 0, 0
    #                     # joystick_x, joystick_y = self.joystick_to_mobile_base(
    #                     #     controller.joystick_x, controller.joystick_y
    #                     # )
    #                     # if self.joystick_mode[MODE] == 0:
    #                     #     x, y = joystick_x, joystick_y
    #                     # else:
    #                     #     x = joystick_x
    #                     #     theta = joystick_y * 100
    #                     # self.robot.move_mobile_base(x, y, theta)

    #                 elif self.mode == 1:
    #                     head_orientation = self.controller_to_head_orientation(pose, controller.arm)
    #                     self.robot.move_head(head_orientation)
    #                     self.head_previous_pose = head_orientation
    #                     antenna = self.joystick_to_antenna(controller.joystick_x, controller.arm)
    #                     self.robot.move_antenna(antenna)
    #                     self.antenna_previous_position[controller.arm] = antenna
    #                 self.controller_previous_pose[controller.arm] = pose

    #             else:
    #                 raise ValueError(f"Invalid mode: {MODE}, available modes: {DUAL_ARM}, {LEFT_ARM}, {RIGHT_ARM}")

    #         if self.tracker_type == TrackerType.RGBD:
    #             color_frame = self.computer_vision.camera.color_frame[0]
    #             frame = self.computer_vision.visualization_landmarks(
    #                 color_frame,
    #                 self.controller_previous_pose[LEFT_ARM],
    #                 self.controller_previous_pose[RIGHT_ARM],
    #                 0,
    #                 0,
    #                 0,
    #                 text_on=True,
    #             )
    #             cv2.imshow("Color Viewer", frame)
    #             cv2.waitKey(1)
    #         time.sleep(max(0, 1 / frequency - (time.time() - t)))


if __name__ == "__main__":
    teleoperation = Teleoperation(TrackerType.RGBD, mirror_mode=True)
    try:
        teleoperation.teleoperation()

    except KeyboardInterrupt:
        for controller in teleoperation.controllers.values():
            controller.stop()
        teleoperation.robot.stop()
