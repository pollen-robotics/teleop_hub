import threading
import time

import numpy as np
from controllers.so_arm_controller import SoArmController  # type: ignore
from pynput import keyboard  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from scipy.spatial.transform import Slerp
from teleoperation_modes.teleoperation_base import Teleoperation  # type: ignore


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
            "r_arm": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["r_arm"]),
            "l_arm": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["l_arm"]),
            "head": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["head"]),
            "mobile_base": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["mobile_base"]),
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
        start_orientation = R.from_matrix(self.robot_previous_pose[self.robot_part][:3, :3])
        so_pose = self.so_arm_controller.get_controller_pose()

        pose = self.find_robot_pose(so_pose, self.robot_part, self.top_grasp[self.robot_part])
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
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
                    self.init_tg = True
                    self.top_grasp[self.robot_part] = not self.top_grasp[self.robot_part]

            elif key.char == "h":
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
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
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
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
                rot1 = R.from_euler("xyz", [0, 0, -np.pi / 4], degrees=False).as_matrix()
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
        diff_orientation = R.from_matrix(diff_orientation).as_euler("xyz", degrees=False)
        diff_orientation *= self.orientation_coeff
        if self.mirror:
            diff_orientation[0] = -diff_orientation[0]
            diff_orientation[2] = -diff_orientation[2]
        diff_orientation = R.from_euler("xyz", diff_orientation, degrees=False).as_matrix()
        robot_pose[:3, :3] = diff_orientation @ self.robot_previous_pose[arm][:3, :3]

        self.so_previous_pose[arm] = so_pose

        return robot_pose

    def trigger_joint_to_gripper_joint(self, joint, arm):
        joint = np.rad2deg(joint)
        min_joint, max_joint = -60, -15
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper

        return int(gripper_opening)

    def trigger_joint_to_antenna_joint(self, joint):
        min_joint, max_joint = -60, 0
        min_antenna, max_antenna = 30, -160
        if joint < -60:
            joint = -60
        if joint > -0:
            joint = -0
        antenna_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_antenna - min_antenna) + min_antenna

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

        orientation_matrix = R.from_euler("xyz", [head_roll, head_pitch, head_yaw], degrees=True).as_matrix()

        return orientation_matrix

    def so_pose_to_mobile_base_speed(self, so_pose):
        diff_position = so_pose[:3, 3] - self.so_previous_pose["mobile_base"][:3, 3]

        x = diff_position[0] * 5
        y = diff_position[1] * 5
        diff_orientation = so_pose[:3, :3] @ self.so_previous_pose["mobile_base"][:3, :3].T
        diff_orientation = R.from_matrix(diff_orientation).as_euler("xyz", degrees=False)

        theta = diff_orientation[2]
        theta = np.rad2deg(theta)

        return x, y, theta
