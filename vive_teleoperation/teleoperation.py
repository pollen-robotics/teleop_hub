import time
from controller.controller import Controller
from robots.reachy2 import Reachy2
import numpy as np

DUAL_ARM = "dual_arm"
LEFT_ARM = "l_arm"
RIGHT_ARM = "r_arm"
MODE = RIGHT_ARM


class Teleoperation:
    def __init__(self):
        self.stop = True

        if MODE == DUAL_ARM:
            self.controllers = {
                LEFT_ARM: Controller(LEFT_ARM),
                RIGHT_ARM: Controller(RIGHT_ARM),
            }
        elif MODE == LEFT_ARM:
            self.controllers = {LEFT_ARM: Controller(LEFT_ARM)}
        elif MODE == RIGHT_ARM:
            self.controllers = {RIGHT_ARM: Controller(RIGHT_ARM)}
        else:
            raise ValueError(
                f"Invalid mode: {MODE}, available modes: {DUAL_ARM}, {LEFT_ARM}, {RIGHT_ARM}"
            )

        self.robot = Reachy2()

        self.controller_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}

        self.robot_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}

        self.stop_count = 0
        self.mode = 0
        self.joystick_mode = {LEFT_ARM: 0, RIGHT_ARM: 0}

        self.controller_previous_button = {RIGHT_ARM: [1, 1, 1], LEFT_ARM: [1, 1, 1]}

        # self.head_previous_pose = np.eye(3)

        # self.antenna_previous_position = 0

        # for controller in self.controllers.values():
        #     self.controller_previous_pose[controller.arm] = controller.get_controller_pose()
        #     self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)

    def init_teleoperation(self):
        self.robot.unfreeze("r_arm")
        self.robot.init_robot()
        self.head_previous_pose = np.eye(3)
        self.antenna_previous_position = {RIGHT_ARM: 0, LEFT_ARM: 0}

        for controller in self.controllers.values():
            controller.init_controller()
            self.controller_previous_pose[
                controller.arm
            ] = controller.get_controller_pose()

            self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)

    def controller_pose_to_robot_pose(self, controller_pose, arm):
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
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (
            max_gripper - min_gripper
        ) + min_gripper
        return gripper_opening

    def controller_to_head_orientation(self, controller_pose, arm):
        diff_orientation = (
            controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T
        )

        head_orientation = diff_orientation @ self.head_previous_pose
        return head_orientation

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
        if MODE == DUAL_ARM:
            if self.controllers[LEFT_ARM].buttonB == 0:
                self.stop_count += 1
            else:
                self.stop_count = 0
            if self.stop_count == 100:
                self.stop = not self.stop
                if not self.stop:
                    self.init_teleoperation()
                else:
                    self.robot.stop()

            if (
                self.controllers[RIGHT_ARM].buttonA == 0
                and self.controllers[RIGHT_ARM].buttonA
                != self.controller_previous_button[RIGHT_ARM][1]
            ):
                print("Change mode")
                if self.mode == 1:
                    self.mode = 0
                else:
                    self.mode = 1

            if self.controllers[LEFT_ARM].buttonA == 0:
                self.mode = 2
            else:
                if not self.mode == 1:
                    self.mode = 0

            if (
                self.controllers[RIGHT_ARM].joystick_button == 0
                and self.controllers[RIGHT_ARM].joystick_button
                != self.controller_previous_button[RIGHT_ARM][0]
            ):
                print("Change joystick mode")
                if self.joystick_mode[RIGHT_ARM] == 1:
                    self.joystick_mode[RIGHT_ARM] = 0
                else:
                    self.joystick_mode[RIGHT_ARM] = 1
            if (
                self.controllers[LEFT_ARM].joystick_button == 0
                and self.controllers[LEFT_ARM].joystick_button
                != self.controller_previous_button[LEFT_ARM][0]
            ):
                print("Change joystick mode")
                if self.joystick_mode[LEFT_ARM] == 1:
                    self.joystick_mode[LEFT_ARM] = 0
                else:
                    self.joystick_mode[LEFT_ARM] = 1

            for controller in self.controllers.values():
                self.controller_previous_button[controller.arm] = [
                    controller.joystick_button,
                    controller.buttonA,
                    controller.buttonB,
                ]
        if MODE == RIGHT_ARM or MODE == LEFT_ARM:
            if self.controllers[MODE].buttonB == 0:
                self.stop_count += 1
            else:
                self.stop_count = 0
            if self.stop_count == 100:
                self.stop = not self.stop
                if not self.stop:
                    self.init_teleoperation()
                else:
                    self.robot.stop()

            if (
                self.controllers[MODE].buttonA == 0
                and self.controllers[MODE].buttonA
                != self.controller_previous_button[MODE][1]
            ):
                print("Change mode")
                if self.mode == 1:
                    self.mode = 0
                else:
                    self.mode = 1

            if (
                self.controllers[MODE].joystick_button == 0
                and self.controllers[MODE].joystick_button
                != self.controller_previous_button[MODE][0]
            ):
                print("Change joystick mode")
                self.robot.unfreeze(MODE)
                self.robot_previous_pose[MODE] = self.robot.fk(MODE)
                if self.joystick_mode[MODE] == 1:
                    self.joystick_mode[MODE] = 0
                else:
                    self.joystick_mode[MODE] = 1

            self.controller_previous_button[MODE] = [
                self.controllers[MODE].joystick_button,
                self.controllers[MODE].buttonA,
                self.controllers[MODE].buttonB,
            ]

    def teleoperation(self):
        frequency = 100
        while True:
            t = time.time()
            self.manage_mode()
            if not self.stop:
                if MODE == DUAL_ARM:
                    if self.mode == 2:
                        for controller in self.controllers.values():
                            pose = controller.get_controller_pose()
                            self.controller_previous_pose[controller.arm] = pose
                            self.robot.unfreeze(controller.arm)
                            self.robot_previous_pose[controller.arm] = self.robot.fk(
                                controller.arm
                            )
                    else:
                        for controller in self.controllers.values():
                            pose = controller.get_controller_pose()

                            # print(pose[:3, :3])
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
                elif MODE == RIGHT_ARM or MODE == LEFT_ARM:
                    controller = self.controllers[MODE]
                    pose = controller.get_controller_pose()
                    if self.mode == 0:
                        robot_pose = self.controller_pose_to_robot_pose(
                            pose, controller.arm
                        )
                        self.robot.go_to_pose(robot_pose, controller.arm)
                        self.robot_previous_pose[controller.arm] = robot_pose
                        x, y, theta = 0, 0, 0
                        joystick_x, joystick_y = self.joystick_to_mobile_base(
                            controller.joystick_x, controller.joystick_y
                        )
                        if self.joystick_mode[MODE] == 0:
                            x, y = joystick_x, joystick_y
                        else:
                            x = joystick_x
                            theta = joystick_y * 100
                        self.robot.move_mobile_base(x, y, theta)
                    elif self.mode == 1:
                        head_orientation = self.controller_to_head_orientation(
                            pose, controller.arm
                        )
                        self.robot.move_head(head_orientation)
                        self.head_previous_pose = head_orientation
                        antenna = self.joystick_to_antenna(
                            controller.joystick_x, controller.arm
                        )
                        self.robot.move_antenna(antenna)
                        self.antenna_previous_position[controller.arm] = antenna
                    self.controller_previous_pose[controller.arm] = pose
                else:
                    raise ValueError(
                        f"Invalid mode: {MODE}, available modes: {DUAL_ARM}, {LEFT_ARM}, {RIGHT_ARM}"
                    )
            time.sleep(max(0, 1 / frequency - (time.time() - t)))


if __name__ == "__main__":
    teleoperation = Teleoperation()
    try:
        teleoperation.teleoperation()

    except KeyboardInterrupt:
        for controller in teleoperation.controllers.values():
            controller.stop()
        teleoperation.robot.stop()
