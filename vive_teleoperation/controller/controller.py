import time
import threading

from controller.arduino import ArduinoController
from controller.feetech import Feetech
from controller.vive_tracker import ViveTracker

import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import make_homogenous_matrix_from_rotation_matrix


feetech_ports = {
    "l_arm": "/dev/noVR_left_motor",
    "r_arm": "/dev/noVR_right_motor",
}

arduino_ports = {
    "l_arm": "/dev/noVR_left_arduino",
    "r_arm": "/dev/noVR_right_arduino",
}

gripper_joints = {
    "l_arm" : [65, 30],
    "r_arm" : [-65, -30]
}


class Controller:
    def __init__(self, arm):
        self.arduino = ArduinoController(arduino_ports[arm])
        # self.feetech = Feetech(feetech_ports[arm])
        self.tracker = ViveTracker(arm)

        self.stop_flag = False
        self.arm = arm

        self.joystick_x = None
        self.joystick_y = None
        self.joystick_button = None
        self.buttonA = None
        self.buttonB = None

        thread = threading.Thread(target=self._update_arduino_data)
        thread.daemon = True    
        thread.start()

        self.gripper_joints_limit = gripper_joints[arm]
        # self.init_gripper(self.gripper_joints_limit[1])

        self.tracker.update_tracker_pose()
        self.tracker_init_pose = self.tracker.tracker_pose




    # def init_gripper(self, joint):
    #     self.feetech.enable_torque()
    #     self.feetech.goto_joints([joint], 1)
    #     self.feetech.disable_torque()

    def init_controller(self):
        self.tracker.update_tracker_pose()
        self.tracker_init_pose = self.tracker.tracker_pose


    def _update_arduino_data(self):
        while not self.stop_flag:
            x, y, button_cmd, buttonA, buttonB = self.arduino.read()
            if x is not None:
                self.joystick_x = x
                self.joystick_y = y
                self.joystick_button = button_cmd
                self.buttonA = buttonA
                self.buttonB = buttonB
            else:
                print("No data")
            time.sleep(0.1)

    # def get_gripper_joint(self):
    #     return self.feetech.get_joints()[0]
    
    def get_controller_pose(self):
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose
        pose = self.rotate_pose(pose, self.arm)
        return pose
    
    def rotate_pose(self, pose, arm):
        
        relative_pose = np.linalg.inv(self.tracker_init_pose) @ pose

        if arm == "r_arm":
            angle_rotation = -30
        else:
            angle_rotation = 30
        rotation = R.from_euler("xyz", [0, 0, angle_rotation], degrees=True).as_matrix()
        Trot = make_homogenous_matrix_from_rotation_matrix(
            rotation, [0, 0, 0])
        relative_pose = Trot @ relative_pose


        if arm == "r_arm":
            rotation = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
            Trot = make_homogenous_matrix_from_rotation_matrix(
                rotation, [0, 0, 0])
            relative_pose = Trot @ relative_pose

        if arm == "l_arm":
            rotation = R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
            Trot = make_homogenous_matrix_from_rotation_matrix(
                rotation, [0, 0, 0])
            relative_pose = Trot @ relative_pose

        return relative_pose
        
    def stop(self):
        self.stop_flag = True
        self.arduino.close()
        # self.feetech.close()


if __name__ == "__main__":
    controller = Controller("r_arm")
    # time.sleep(0.5)

    frequency = 100
    try :
        while True:
            print(f"x: {controller.joystick_x}, y: {controller.joystick_y}, button_cmd: {controller.joystick_button}, buttonA: {controller.buttonA}, buttonB: {controller.buttonB}")
            gripper_joint = controller.get_gripper_joint()
            print(f"Gripper joint: {gripper_joint}")
            pose = controller.get_controller_pose()
            print(f"Controller pose: {pose}")
            time.sleep(1/frequency)
            
    except KeyboardInterrupt:
        controller.stop()
   
   
