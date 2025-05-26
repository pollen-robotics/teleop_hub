import time

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore
from teleoperation_mode.teleoperation import LEFT_ARM, RIGHT_ARM, Teleoperation
from utils import axis_cleaner, parse_hat


class GamepadTeleoperation(Teleoperation):
    """Teleoperation using a Gamepad controller, driving Reachy2 with the generic robot interface."""

    def __init__(self, angle_step, x_y_joystick_ratio, z_increment) -> None:
        super().__init__()
        import pygame  # type: ignore

        self.pygame = pygame

        # Initialize pygame joystick
        self.pygame.init()
        self.pygame.joystick.init()
        if self.pygame.joystick.get_count() == 0:
            raise RuntimeError("No Gamepad controller found")
        self.joystick = self.pygame.joystick.Joystick(0)
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
        """Read Gamepad inputs, update robot target poses, grippers, and send commands."""
        self.pygame.event.pump()

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
                self.robot.move_gripper(RIGHT_ARM, True, 130 * self.gripper_state[RIGHT_ARM])
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
                self.robot.move_gripper(LEFT_ARM, True, 130 * self.gripper_state[LEFT_ARM])
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
            rot_delta = R.from_euler("xyz", [delta["roll"], delta["pitch"], delta["yaw"]], degrees=True).as_matrix()
            new_pose[:3, :3] = rot_delta.dot(prev[:3, :3])
            # send command
            self.robot.go_to_pose(new_pose, arm)
            # update for next iteration
            self.robot_previous_pose[arm] = new_pose

        if self._stop_flag:
            self.stop_flag = True
