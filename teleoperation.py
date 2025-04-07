import time

import cv2  # type: ignore

from controller.camera import Camera
from controller.controller import Controller
from reachy import Reachy2

DUAL_ARM = "dual_arm"
LEFT_ARM = "l_arm"
RIGHT_ARM = "r_arm"
MODE = DUAL_ARM


class Teleoperation:
    def __init__(self, marker_size=0.04):
        self.camera = Camera()

        if MODE == DUAL_ARM:
            self.controllers = {
                LEFT_ARM: Controller(LEFT_ARM, marker_size, self.camera),
                RIGHT_ARM: Controller(RIGHT_ARM, marker_size, self.camera),
            }
        elif MODE == LEFT_ARM:
            self.controllers = {LEFT_ARM: Controller(LEFT_ARM, marker_size, self.camera)}
        elif MODE == RIGHT_ARM:
            self.controllers = {RIGHT_ARM: Controller(RIGHT_ARM, marker_size, self.camera)}
        else:
            raise ValueError(f"Invalid mode: {MODE}, available modes: {DUAL_ARM}, {LEFT_ARM}, {RIGHT_ARM}")

        self.robot = Reachy2()
        self.robot.init_robot()

        self.controller_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}

        self.robot_previous_pose = {LEFT_ARM: None, RIGHT_ARM: None}

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

    def teleoperation(self):
        frequency = 100
        while True:
            t = time.time()
            for controller in self.controllers.values():

                pose = controller.get_controller_pose()
                robot_pose = self.controller_pose_to_robot_pose(pose, controller.arm)
                self.robot.go_to_pose(robot_pose, controller.arm)
                self.controller_previous_pose[controller.arm] = pose
                self.robot_previous_pose[controller.arm] = robot_pose
            frame = self.camera.get_frame_with_cube_pose(
                [self.controllers[arm].aruco_cube.cube_pose for arm in self.controllers.keys()]
            )
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            time.sleep(max(0, 1 / frequency - (time.time() - t)))


if __name__ == "__main__":
    teleoperation = Teleoperation()
    try:
        teleoperation.teleoperation()

    except KeyboardInterrupt:
        for controller in teleoperation.controllers.values():
            controller.stop()
        teleoperation.robot.stop()
