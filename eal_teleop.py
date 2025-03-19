import time
import sys
import numpy as np
import math
import numpy.typing as npt
import triad_openvr
import openvr

from reachy2_sdk import ReachySDK  # type: ignore

from reachy2_sdk_api.kinematics_pb2 import Matrix4x4  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from scipy.spatial.transform import Slerp  # type: ignore

import threading
from pynput import keyboard
from mobile_base_controller import MobileBaseController, JoystickData

from teleop_tracker import RobotController


from utils import make_homogenous_matrix_from_rotation_matrix, create_plot, plot_orientation, update_plot, fk


arduino_ports = {
    "l_arm": "/dev/noVR_left_arduino",
    "r_arm": "/dev/noVR_right_arduino",
}


class Teleop:
    def __init__(self, ip):
        
        self.reachy = ReachySDK(ip)
        position = [0.36, -0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        r_pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        position = [0.36, 0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        l_pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        self.reachy_previous_pose = {"r_arm" : r_pose, "l_arm" : l_pose}
        self.reachy_head_joints = [0, 0, 0]


        self.init_reachy()

        self.trackers = {
            "l_arm" : RobotController("l_arm", self.reachy),
            "r_arm" : RobotController("r_arm", self.reachy),
        }

        # time.sleep(1)
        for tracker in self.trackers.values():
            tracker.gripper.disable_torque()


        self.mobile_base = MobileBaseController(self.reachy, left_port_joystick=arduino_ports["l_arm"], right_port_joystick=arduino_ports["r_arm"], two_trackers_mode=True)
        self.command_getter = threading.Thread(target=self.mobile_base.run, daemon=True)
        self.command_getter.start()


        self.reachy.reset_default_limits()
        # keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        # keyboard_thread.start()


    def init_reachy(self):
        self.reachy.turn_on()
        self.reachy.mobile_base.reset_odometry()
        self.reachy.head.l_antenna.turn_on()
        self.reachy.head.r_antenna.turn_on()
        self.reachy.r_arm.gripper.open()
        self.reachy.l_arm.gripper.open()
        joints = self.reachy.r_arm.inverse_kinematics(self.reachy_previous_pose["r_arm"])
        self.reachy.r_arm.goto(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
        joints = self.reachy.l_arm.inverse_kinematics(self.reachy_previous_pose["l_arm"])
        self.reachy.l_arm.goto(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
        self.reachy.head.goto(self.reachy_head_joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)

    def teleoperation(self):

        stop_pressed = 0
        frequency = 100
        while True:
            t = time.time()
            self.trackers["r_arm"].tracker.update_tracker_pose()
            self.trackers["r_arm"].update_control_arm("r_arm")
            self.trackers["l_arm"].tracker.update_tracker_pose()
            self.trackers["l_arm"].update_control_arm("l_arm")
            # if self.mobile_base.stop:
            # print(self.mobile_base.stop)
            if self.mobile_base.stop:
                stop_pressed += 1
            else :
                stop_pressed = 0
            if stop_pressed > 100:
                break
            time.sleep(max(0, 1/frequency - (time.time() - t)))



    def on_press(self, key):
        print(f'Key {key} pressed')
        try:
            if key.char == 'l':
                print("press l")
                if not self.init and not self.pause and self.reachy_part != "l_arm" and not self.mouse and not self.init_tg:
                    self.init = True
                    self.reachy_part = "l_arm"
                
            elif key.char == 'r':
                if not self.init and not self.pause and self.reachy_part != "r_arm" and not self.mouse and not self.init_tg:
                    self.init = True
                    self.reachy_part = "r_arm"

            elif key.char == 'o':
                if not self.init and not self.pause:
                    self.mouse = not self.mouse

            elif key.char == 'i':
                self.position_coeff += 0.1
                print(f'Position coeff: {self.position_coeff}')

            elif key.char == 'k':
                self.position_coeff -= 0.1
                print(f'Position coeff: {self.position_coeff}')

            elif key.char == 'u':
                self.orientation_coeff += 0.1
                print(f'Orientation coeff: {self.orientation_coeff}')

            elif key.char == 'j':
                self.orientation_coeff -= 0.1
                print(f'Orientation coeff: {self.orientation_coeff}')

            elif key.char == 'n':
                self.mirror = not self.mirror
                print(f'Mirror: {self.mirror}')

            

        except AttributeError:
            print(f'Key {key} pressed')

    def listen_keyboard(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()





if __name__ == "__main__":
    try:

        robot_ip = "192.168.10.109"
        # robot_ip = "localhost"
        teleop = Teleop(robot_ip)
        # time.sleep(1)

        teleop.teleoperation()
        
        teleop.reachy.turn_off_smoothly()
        for tracker in teleop.trackers.values():
            tracker.gripper.close()
        
    except KeyboardInterrupt:
        teleop.reachy.turn_off_smoothly()
        for tracker in teleop.trackers.values():
            tracker.gripper.close()
        print("Exiting...")
        exit(0)
    # try:
    #     # teleop = RobotController("/dev/ttyACM0", "192.168.10.107")
    #     side = "l_arm"
    #     robot_ip = "192.168.10.109"
    #     # robot_ip = "localhost"
    #     teleops = {
    #         "l_arm" : RobotController("l_arm", robot_ip),
    #         "r_arm" : RobotController("r_arm", robot_ip),
    #     }

    #     mobile_base = MobileBaseController(teleops["l_arm"].reachy, port_joystick=arduino_ports["l_arm"], port_joystick_2=arduino_ports["r_arm"], two_trackers_mode=True)

    #     # thread = threading.Thread(target=mobile_base.run)
    #     # thread.start()
    #     # teleop = RobotController(side, "localhost")
    #     frequency = 100
    #     time.sleep(1)
    #     fig, ax = create_plot()

    #     while True:
    #         # print("_______")
    #         t = time.time()
    #         # part = teleop.reachy_part

    #         if teleops["r_arm"].mouse:
    #             pass
    #             # joints = teleop.so100.get_joints()
    #             # pose = teleop.tracker.tracker_pose

    #             # pose = teleop.find_reachy_pose(fk(joints), part, teleop.top_grasp[part])
    #             # teleop.so_previous_joints[part] = joints
    #         elif teleops["r_arm"].init:
    #             print("switching arm")
    #             teleops["r_arm"].init_so()
    #             print("done")
    #             teleops["r_arm"].init = False
    #         else :
    #             # print("ici")
    #             teleops["r_arm"].tracker.update_tracker_pose()
    #             teleops["r_arm"].update_control_arm("r_arm")
    #             teleops["l_arm"].tracker.update_tracker_pose()
    #             teleops["l_arm"].update_control_arm("l_arm")


    #         # print(max(0, 1/frequency - (time.time() - t)))
            # time.sleep(max(0, 1/frequency - (time.time() - t)))
    # except KeyboardInterrupt:
    #     teleops["r_arm"].so100.close()
    #     teleops["r_arm"].reachy.turn_off_smoothly()
    #     print("Exiting...")
    #     exit(0)
