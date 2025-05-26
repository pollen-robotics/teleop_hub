import threading
import time

import numpy as np
from controllers.so_arm_controller import SoArmController  # type: ignore
from pynput import keyboard  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from scipy.spatial.transform import Slerp
from teleoperation_modes.teleoperation_base import Teleoperation  # type: ignore


class SoArmTeleoperation(Teleoperation):
    """Teleoperation class with SO-100 controller."""

    def __init__(self, port: str) -> None:
        """Initialize the SoArmTeleoperation.

        Args:
            port (str): The serial port to which the SO-100 is connected.
        """
        super().__init__()

        self.so_arm_controller = SoArmController(port)

        time.sleep(1)
        self.robot_part = "r_arm"
        self.init = False
        self.pause = False
        self.mouse = False
        self.mobile_base = False
        self.mirror = False

        self.top_grasp = {"r_arm": False, "l_arm": False}
        self.init_tg = False

        self.position_coeff = 1
        self.orientation_coeff = 1

        # Initialize the previous poses for the SO-100 controller.
        self.so_previous_pose = {
            "r_arm": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["r_arm"]),
            "l_arm": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["l_arm"]),
            "head": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["head"]),
            "mobile_base": self.so_arm_controller.so_arm_fk(self.so_arm_controller.so_previous_joints["mobile_base"]),
        }

    def init_teleoperation(self) -> None:
        """Initialize the teleoperation.

        Initializes the robot, sets the initial robot part, and starts the keyboard listener thread.
        """
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

    def step(self) -> None:
        """Step called at each iteration of the teleoperation loop.

        This method updates the robot's pose based on the SO-100 controller's pose
        and the current robot part. It handles different robot parts (arms, head, mobile base)
        and applies transformations based on the controller pose.
        """
        part = self.robot_part

        if part == "l_arm" or part == "r_arm":
            top_grasp = self.top_grasp[part]

        if self.pause:
            self.robot.move_mobile_base(x=0, y=0, theta=0)

        elif self.mouse:
            self.so_arm_controller.update_joints(part)
            self.so_previous_pose[part] = self.so_arm_controller.get_controller_pose()

        elif self.init:
            self.so_arm_controller.init_controller(part)
            self.init = False

        elif self.init_tg:
            self.init_top_grasp()
            self.init_tg = False

        elif part == "head":
            so_joints = self.so_arm_controller.get_head_joints()
            head_joints = self.so_joints_to_head_joints(so_joints)
            self.robot.move_head(head_joints)
            trigger_joint = self.so_arm_controller.get_gripper_joint()
            antenna_joint = self.trigger_joint_to_gripper_joint(trigger_joint)
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
            trigger_joint = self.trigger_joint_to_gripper_joint(controller_joint)
            self.robot.move_gripper(part, True, trigger_joint)

    def init_top_grasp(self) -> None:
        """Initialize the top grasp for the current robot part.

        Put the arm in the "top grasp" position, i.e. the gripper pointing downwards
        with the elbow raised. It performs a spherical linear interpolation (SLERP)
        between the current pose and the target pose for a smooth transition.
        """
        self.so_arm_controller.lock_arm()
        start_orientation = R.from_matrix(self.robot_previous_pose[self.robot_part][:3, :3])
        so_pose = self.so_arm_controller.get_controller_pose()

        pose = self.find_robot_pose(so_pose, self.robot_part, self.top_grasp[self.robot_part])
        self.robot_previous_pose[self.robot_part] = pose
        end_orientation = R.from_matrix(pose[:3, :3])

        frequency = 100
        duration = 2.0
        steps = int(duration * frequency)

        slerp = Slerp([0, 1], R.concatenate([start_orientation, end_orientation]))

        # Perform spherical linear interpolation (SLERP)
        for i in range(steps):
            alpha = i / steps
            interpolated_rotation = slerp([alpha])
            pose[:3, :3] = interpolated_rotation.as_matrix()[0]
            self.robot.go_to_pose(pose, self.robot_part)
            time.sleep(1 / frequency)
        self.robot_previous_pose[self.robot_part] = pose
        self.so_arm_controller.unlock_arm()

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Handle key press events.
        This method is called when a key is pressed. It checks the key and performs
        the corresponding action based on the current state of the teleoperation.
            - l : controls the left arm
            - r : controls the right arm
            - h : controls the head
            - p : pauses the teleoperation
            - o : stops the robot's arm while allowing the SO-100 to be moved
                    to change reference frame
            - t : toggles the top grasp for the current arm
            - i /k : increases/ decreases the position coefficient
            - u / j : increases/ decreases the orientation coefficient
            - m : controls the mobile base
            - n : toggles the mirror mode

        Args:
            key (keyboard.Key | keyboard.KeyCode): The key that was pressed.
        """
        print(f"Key {key} pressed")

        try:
            # to control left arm
            if key.char == "l":
                if (
                    not self.init
                    and not self.pause
                    and self.robot_part != "l_arm"
                    and not self.mouse
                    and not self.init_tg
                ):
                    print("Switching to left arm control")
                    self.init = True
                    self.robot_part = "l_arm"

            # to control right arm
            elif key.char == "r":
                if (
                    not self.init
                    and not self.pause
                    and self.robot_part != "r_arm"
                    and not self.mouse
                    and not self.init_tg
                ):
                    print("Switching to right arm control")
                    self.init = True
                    self.robot_part = "r_arm"

            # to pause the teleoperation
            elif key.char == "p":
                if not self.init and not self.init_tg:
                    self.pause = not self.pause
                    if self.pause:
                        print("Teleoperation paused")
                        self.so_arm_controller.lock_arm()
                    else:
                        print("Teleoperation resumed")
                        self.so_arm_controller.unlock_arm()

            # to stop the robot's arm while allowing the SO-100 to be moved
            elif key.char == "o":
                if not self.init and not self.pause:
                    print("Stopping the robot until another 'o' command")
                    self.mouse = not self.mouse

            # to toggle the top grasp for the current arm
            elif key.char == "t":
                print("Switching to top grasp mode")
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
                    self.init_tg = True
                    self.top_grasp[self.robot_part] = not self.top_grasp[self.robot_part]

            # to control the head
            elif key.char == "h":
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
                    print("Switching to head control")
                    self.init = True
                    self.robot_part = "head"

            # to increase the position coefficient
            elif key.char == "i":
                self.position_coeff += 0.1
                print(f"Position coeff: {round(self.position_coeff,2)}")

            # to decrease the position coefficient
            elif key.char == "k":
                self.position_coeff -= 0.1
                print(f"Position coeff: {round(self.position_coeff,2)}")

            # to increase the orientation coefficient
            elif key.char == "u":
                self.orientation_coeff += 0.1
                print(f"Orientation coeff: {round(self.orientation_coeff,2)}")

            # to decrease the orientation coefficient
            elif key.char == "j":
                self.orientation_coeff -= 0.1
                print(f"Orientation coeff: {round(self.orientation_coeff,2)}")

            # to control the mobile base
            elif key.char == "m":
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
                    print("Switching to mobile base control")
                    self.init = True
                    self.robot_part = "mobile_base"

            # to toggle the mirror mode
            elif key.char == "n":
                self.mirror = not self.mirror
                print(f"Mirror mode: {self.mirror}")

        except AttributeError:
            print(f"Key {key} pressed")

    def listen_keyboard(self) -> None:
        """Listen for keyboard events.

        This method starts a keyboard listener that listens for key presses and calls the `on_press` method.
        It runs in a separate thread to allow the teleoperation loop to continue running.
        """
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def find_robot_pose(self, so_pose: list[float], arm: str, top_grasp: bool) -> np.ndarray:
        """Find the robot pose based on the SO-100 controller's pose.

        This method calculates the robot's pose based on the SO-100 controller's pose
        and the current robot part. It applies transformations to the pose based on the
        controller pose and the current arm.

        Args:
            so_pose (list[float]): The pose of the SO-100 controller as a 4x4 transformation matrix.
            arm (str): The current robot part, can be "r_arm", "l_arm", "head", or "mobile_base".
            top_grasp (bool): Whether the current arm is in the top grasp position.
        Returns:
            np.ndarray: The robot pose as a 4x4 transformation matrix.
        """
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

    def trigger_joint_to_gripper_joint(self, joint: float) -> int:
        """Convert the trigger joint value to the gripper opening angle.

        The gripper opening angle is calculated based on the joint value, which is expected to be in degrees.
        The joint value is mapped to a gripper opening angle between 0 and 130 degrees.

        Args:
            joint (float): The joint value in degrees.
        Returns:
            int: The gripper opening angle in degrees.
        """
        joint = np.rad2deg(joint)
        min_joint, max_joint = -60, -15
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper

        return int(gripper_opening)

    def trigger_joint_to_antenna_joint(self, joint: float) -> int:
        """Convert the trigger joint value to the antenna opening angle.

        The antenna opening angle is calculated based on the joint value, which is expected to be in degrees.
        The joint value is mapped to an antenna opening angle between -160 and 30 degrees.

        Args:
            joint (float): The joint value in degrees.
        Returns:
            int: The antenna opening angle in degrees.
        """
        min_joint, max_joint = -60, 0
        min_antenna, max_antenna = 30, -160
        if joint < -60:
            joint = -60
        if joint > -0:
            joint = -0
        antenna_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_antenna - min_antenna) + min_antenna

        return int(antenna_opening)

    def so_joints_to_head_joints(self, joints: list[float]) -> np.ndarray:
        """Convert the SO-100 controller's joints to the robot's head joints.

        This method converts the SO-100 controller's joints (roll, pitch, yaw) to the robot's head joints.
        The SO-100 controller's joints are expected to be in radians, and the robot's head joints
        are expected to be in degrees. The head roll and pitch are mapped to a specific range,
        and the head yaw is adjusted by a constant offset.

        Args:
            joints (list[float]): The SO-100 controller's joints as a list of floats.
                Expected to be in radians: [roll, pitch, yaw].
        Returns:
            np.ndarray: The robot's head joints as a 3x1 numpy array in degrees.
        """
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

    def so_pose_to_mobile_base_speed(self, so_pose: np.ndarray) -> tuple[float, float, float]:
        """
        Convert the SO-100 controller's pose to mobile base speed.

        Args:
            so_pose (np.ndarray): The 4x4 pose of the SO-100 controller.

        Returns:
            tuple[float, float, float]: Speed in x, y (m/s), and theta (deg/s).
        """
        # Position difference
        prev_pose = self.so_previous_pose["mobile_base"]
        diff_pos = so_pose[:3, 3] - prev_pose[:3, 3]
        norm_diff = np.linalg.norm(diff_pos[:2])

        # Compute linear speeds (x, y)
        if norm_diff < 0.03:
            x_speed, y_speed = 0.0, 0.0
        else:
            x_speed = diff_pos[0] * 5.0
            y_speed = diff_pos[1] * 5.0

        # Compute angular speed (theta)
        delta_rot = so_pose[:3, :3] @ prev_pose[:3, :3].T
        delta_euler = R.from_matrix(delta_rot).as_euler("xyz", degrees=False)
        theta = np.degrees(delta_euler[2])

        if abs(theta) < 10:
            theta = 0.0

        return x_speed, y_speed, theta
