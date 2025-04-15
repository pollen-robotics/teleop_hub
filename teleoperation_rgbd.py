import time

import cv2  # type: ignore
import numpy as np

from controller.computer_vision import ComputerVision
from controller.orbbec import Orbbec
from controller.robot_controller import RobotController


class TeleopControl:
    def __init__(
        self, camera: Orbbec, robot_controller: RobotController, scale_percent: int, timestep: float, mirror_mode=False
    ):
        self.camera = camera
        self.vision = ComputerVision(self.camera, scale_percent, up_mode=True)
        self.robot_controller = robot_controller
        self.timestep = timestep
        self.first_pose_done = False
        self.mirror_mode = mirror_mode

    def is_first_command_ok(self, command):
        # check if the first command is in the cube : x 0,3/O,45 y 0,15/0.3 z -0,35/-0.2
        if (
            command[0] < 0.45
            and command[0] > 0.3
            and command[1] < 0.3
            and command[1] > 0.15
            and command[2] < -0.15
            and command[2] > -0.35
        ):
            return True
        return False

    def raise_command_to_stop(self):
        if len(self.robot_controller.head_rpy) == 10:
            pitch_values = [rpy[1] for rpy in self.robot_controller.head_rpy]
            roll_values = [rpy[0] for rpy in self.robot_controller.head_rpy]
            yaw_values = [rpy[2] for rpy in self.robot_controller.head_rpy]
            if (
                np.all(np.abs(pitch_values) > 25)
                and np.all(np.abs(roll_values) < 10)
                and np.all(np.abs(yaw_values) < 10)
            ):
                print("Command to stop")
                return True
        return False

    def run(self):
        self.robot_controller.reachy.goto_posture("elbow_90", wait=True)
        self.robot_controller.former_poses = [
            self.robot_controller.reachy.l_arm.forward_kinematics(),
            self.robot_controller.reachy.r_arm.forward_kinematics(),
        ]

        # first pose
        while not self.first_pose_done:
            try:
                color_frame = self.camera.color_frame.pop()
                depth_frame = self.camera.depth_frame.pop()
                self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)
                if np.any(self.vision.user_center) and np.any(self.vision.wrists):
                    left_goal_pose = self.robot_controller.get_effector_pose(
                        "left", self.vision, "elbow", dist_filter=False, mirror_mode=self.mirror_mode
                    )
                    right_goal_pose = self.robot_controller.get_effector_pose(
                        "right", self.vision, "elbow", dist_filter=False, mirror_mode=self.mirror_mode
                    )

                    if self.is_first_command_ok(left_goal_pose[:3, 3]) and not self.is_first_command_ok(
                        right_goal_pose[:3, 3]
                    ):

                        if self.mirror_mode:
                            left_goal_pose_mirror = self.robot_controller.get_mirror_pose(right_goal_pose)
                            right_goal_pose_mirror = self.robot_controller.get_mirror_pose(left_goal_pose)
                            left_goal_pose = left_goal_pose_mirror
                            right_goal_pose = right_goal_pose_mirror

                        self.robot_controller.make_line([left_goal_pose, right_goal_pose], duration=2)
                        self.first_pose_done = True
                        print("First pose done")

            except IndexError:
                time.sleep(0.005)
                continue

        # teleoperation
        while True:
            t = time.time()
            try:
                color_frame = self.camera.color_frame.pop()
                depth_frame = self.camera.depth_frame.pop()
            except IndexError:
                time.sleep(0.005)
                continue

            self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)

            # make the robot follow the user's hands
            left_goal_pose = self.robot_controller.get_effector_pose("left", self.vision, "elbow", dist_filter=True)
            right_goal_pose = self.robot_controller.get_effector_pose("right", self.vision, "elbow", dist_filter=True)

            if self.mirror_mode:
                left_goal_pose_mirror = self.robot_controller.get_mirror_pose(right_goal_pose)
                right_goal_pose_mirror = self.robot_controller.get_mirror_pose(left_goal_pose)
                left_goal_pose = left_goal_pose_mirror
                right_goal_pose = right_goal_pose_mirror

            self.robot_controller.go_to_pose(left_goal_pose, "l_arm")
            self.robot_controller.go_to_pose(right_goal_pose, "r_arm")

            roll, pitch, yaw = 0, 0, 0
            # make the robot follow the user's head
            if np.any(self.vision.face_points):
                roll, pitch, yaw = self.robot_controller.get_head_rotation(self.vision, self.mirror_mode)
                self.robot_controller.set_head_orientation(roll, pitch, yaw)

            # make the robot grab the objects
            self.robot_controller.get_gripper_command("left", self.vision, self.mirror_mode)
            self.robot_controller.get_gripper_command("right", self.vision, self.mirror_mode)

            # to visualize original_landmarks
            self.vision.visualization_landmarks(
                color_frame,
                left_goal_pose,
                right_goal_pose,
                roll,
                pitch,
                yaw,
                text_on=True,
                body_on=True,
                hands_on=False,
                face_on=True,
            )

            if (time.time() - t) // 10 == 0:
                print(f"Freq: {1/(time.time() - t):.3f}")

            # if esc or ctrl+c is pressed, stop the teleoperation
            if self.raise_command_to_stop() or (cv2.waitKey(1) & 0xFF == 27):
                break

        cv2.destroyAllWindows()
        self.camera.stop()
        self.camera.frame_getter.join(5)


if __name__ == "__main__":
    camera = Orbbec()
    scale_percent = 50
    timestep = 0.02
    robot = RobotController("localhost")
    teleop = TeleopControl(camera, robot, scale_percent, timestep, mirror_mode=True)
    print("Teleoperation started")
    teleop.run()
