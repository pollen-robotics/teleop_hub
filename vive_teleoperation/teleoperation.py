import time
from controller.controller import Controller
from robots.reachy2 import Reachy2
import numpy as np

from scipy.spatial.transform import Rotation as R

DUAL_ARM = "dual_arm"
LEFT_ARM = "l_arm"
RIGHT_ARM = "r_arm"
MODE = DUAL_ARM

class Teleoperation:
    def __init__(self):

        if MODE == DUAL_ARM:
            self.controllers = {
                LEFT_ARM: Controller(LEFT_ARM),
                RIGHT_ARM: Controller(RIGHT_ARM)
            }
        elif MODE == LEFT_ARM:
            self.controllers = {
                LEFT_ARM: Controller(LEFT_ARM)
            }
        elif MODE == RIGHT_ARM:
            self.controllers = {
                RIGHT_ARM: Controller(RIGHT_ARM)
            }
        else:
            raise ValueError(f"Invalid mode: {MODE}, available modes: {DUAL_ARM}, {LEFT_ARM}, {RIGHT_ARM}")
        
        self.robot = Reachy2()
        self.robot.init_robot()

        self.controller_previous_pose = {
            LEFT_ARM: None,
            RIGHT_ARM: None
        }

        self.robot_previous_pose = {
            LEFT_ARM: None,
            RIGHT_ARM: None
        }

        for controller in self.controllers.values():
            self.controller_previous_pose[controller.arm] = controller.get_controller_pose()
            self.robot_previous_pose[controller.arm] = self.robot.fk(controller.arm)



    def controller_pose_to_robot_pose(self, controller_pose, arm):
        robot_pose = self.robot_previous_pose[arm].copy()

        diff_position = controller_pose[:3, 3] - self.controller_previous_pose[arm][:3, 3]
        robot_pose[:3, 3] += diff_position

        diff_orientation = controller_pose[:3, :3] @ self.controller_previous_pose[arm][:3, :3].T 
        robot_pose[:3, :3] = diff_orientation @ self.robot_previous_pose[arm][:3, :3]        
        return robot_pose
    
    def joystick_to_mobile_base(self, x_input, y_input):
        command_max = 0.4
        center_joystick = 512
        x_goal, y_goal = 0, 0

        if not np.isclose(x_input, center_joystick, atol=15):
            y_goal = command_max / center_joystick * x_input - command_max
        if not np.isclose(y_input, center_joystick, atol=15):
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
    
        
    def teleoperation(self):
        frenquency = 100
        while True:
            t = time.time()
            for controller in self.controllers.values():
                pose = controller.get_controller_pose()
                robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm)
                self.robot.go_to_pose(robot_pose, controller.arm)
                self.controller_previous_pose[controller.arm] = pose
                self.robot_previous_pose[controller.arm] = robot_pose

                joint = controller.get_gripper_joint()
                gripper_joint = self.trigger_joint_to_gripper_joint(joint, controller.arm)
                self.robot.move_gripper(gripper_joint, controller.arm)

            if MODE == DUAL_ARM:
                r_x, r_y = self.joystick_to_mobile_base(self.controllers[RIGHT_ARM].joystick_x, self.controllers[RIGHT_ARM].joystick_y)
                l_x, l_y = self.joystick_to_mobile_base(self.controllers[LEFT_ARM].joystick_x, self.controllers[LEFT_ARM].joystick_y)
                if l_x < 0.05 and l_x > -0.05:
                    l_x = 0
                if l_y < 0.05 and l_y > -0.05:
                    l_y = 0
                if r_y < 0.05 and r_y > -0.05:
                    r_y = 0
                theta = r_y * 100
                self.robot.move_mobile_base(l_x, l_y, theta)
            if MODE == RIGHT_ARM:
                x, y = self.joystick_to_mobile_base(self.controllers[RIGHT_ARM].joystick_x, self.controllers[RIGHT_ARM].joystick_y)
                theta = 0
                self.robot.move_mobile_base(x, y, theta)
            time.sleep(max(0, 1/frenquency - (time.time() - t)))


if __name__ == "__main__":
    teleoperation = Teleoperation()

    try:

        teleoperation.teleoperation()

    except KeyboardInterrupt:
        for controller in teleoperation.controllers.values():
            controller.stop()
        teleoperation.robot.stop()



    