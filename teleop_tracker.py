import time
import sys
import numpy as np
import math
import numpy.typing as npt
import triad_openvr
import openvr

from reachy2_sdk import ReachySDK  # type: ignore
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value

from reachy2_sdk.utils.utils import recompose_matrix, get_pose_matrix  # type: ignore
from reachy2_sdk_api.arm_pb2 import (  # type: ignore
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from scipy.spatial.transform import Slerp  # type: ignore

from feetech import Feetech
import threading
from pynput import keyboard




from utils import make_homogenous_matrix_from_rotation_matrix, create_plot, plot_orientation, update_plot, fk

SIDE = "left"

trackers = {
    "l_arm": "LHR-0D914CCE",
    "r_arm": "LHR-D520271F",
}

feetech_ports = {
    "l_arm": "/dev/noVR_left_motor",
    "r_arm": "/dev/noVR_right_motor",
}

arduino_ports = {
    "l_arm": "/dev/noVR_left_arduino",
    "r_arm": "/dev/noVR_right_arduino",
}

gripper_joints = {
    "l_arm" : [60, 15],
    "r_arm" : [-60, -15]
}


class ViveTracker:
    def __init__(self, tracker_id):
        self.tracker_name = tracker_id

        if self.tracker_name not in trackers:
            print(
                f"Tracker name '{self.tracker_name}' not found in trackers dictionary. Available trackers:"
                f" {list(trackers.keys())}"
            )
            sys.exit(1)

        self.vive = triad_openvr.triad_openvr()

        serial_number = trackers[self.tracker_name]
        print(f"Looking for tracker: {self.tracker_name} (Serial: {serial_number})")

        # Find corresponding tracker device in SteamVR
        matched_tracker = None
        for dev in self.vive.devices:
            dev_serial = self.vive.devices[dev].get_serial().decode("utf-8").strip()
            print(f"Found device: {dev} (Serial: {dev_serial})")
            if dev_serial == serial_number:
                matched_tracker = dev
                break

        if matched_tracker is None:
            print(f"Tracker with serial '{serial_number}' not found in SteamVR.")
            sys.exit(1)

        self.tracker_name = matched_tracker

        print(f"Tracker '{tracker_id}' found as '{self.tracker_name}' in SteamVR.")

        self.tracker = self.vive.devices[self.tracker_name]
        self.tracker_euler_angles = np.zeros(3)
        self.tracker_position = np.zeros(3)
        self.tracker_pose = np.eye(4)  # Current pose
        self.zero_pose = None  # Initial reference frame

    def convert_openvr_matrix(self, hmd_matrix):
        """
        Convertit une matrice OpenVR 3x4 (HmdMatrix34_t) en une matrice 4x4 NumPy.

        :param hmd_matrix: Matrice OpenVR (hmd_matrix.mDeviceToAbsoluteTracking)
        :return: Matrice 4x4 NumPy
        """
        m = np.array([
            [hmd_matrix[0][0], hmd_matrix[0][1], hmd_matrix[0][2], hmd_matrix[0][3]],
            [hmd_matrix[1][0], hmd_matrix[1][1], hmd_matrix[1][2], hmd_matrix[1][3]],
            [hmd_matrix[2][0], hmd_matrix[2][1], hmd_matrix[2][2], hmd_matrix[2][3]],
            [0, 0, 0, 1]
        ])
        return m

    def update_tracker_pose(self):
        pose = self.tracker.get_pose_matrix()
        # pose2 = self.tracker.get_pose_euler()
        # print(pose2)

        # pose = np.array(pose)
        # position = pose[:3, 3]
        # orientation = R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True)
        # print(f"position: {position}")
        # print(f"orientation: {orientation}")
        self.tracker_pose = self.convert_openvr_matrix(pose)

    def update_rpy(self):
        [x, y, z, roll, pitch, yaw] = self.tracker.get_pose_euler()
        
        self.tracker_euler_angles = np.array(np.degrees([roll, pitch, yaw]))
        print (self.tracker_euler_angles)


    def get_tracker_position(self) -> npt.ArrayLike:
        return self.tracker_position

    def get_tracker_euler_angles(self) -> npt.ArrayLike:
        return self.tracker_euler_angles

    def calibrate_pose_zero(self):
        print("Calibration : reset zero pose ")
        time.sleep(2)
        self.update_tracker_pose()
        self.zero_pose = self.tracker_pose
        # R_roll = np.array([
        #     [1, 0, 0],
        #     [0, 0, 1],
        #     [0, -1, 0]
        # ])

        # self.center_pose[:3, :3] = self.tracker_pose[:3, :3]

        # print("Calibration : Move the tracker in front of your sternum")
        # user_positions = []
        # time.sleep(2)
        # t0 = time.time()
        # while time.time() - t0 < 3 or len(user_positions) < 10:
        #     user_positions.append(self.tracker.get_pose_euler()[:3])
        #     time.sleep(0.1)
        # mean_position = np.median(user_positions, axis=0)
        
        # self.center_pose[:3,3] = mean_position
        # print("center pose", self.center_pose)


    def get_relative_tracker_pose(self):
        self.update_tracker_pose()
        relative_pose = np.linalg.inv(self.center_pose) @ self.tracker_pose
        relative_position = relative_pose[:3, 3]
        # print("relative_position", relative_position)
        # print("relative orientation in euler angles", R.from_matrix(relative_pose[:3, :3]).as_euler('xyz', degrees=True))
        return relative_pose




class RobotController:
    def __init__(self, arm, ip):


        self.gripper = Feetech(feetech_ports[arm])
        self.reachy = ReachySDK(ip)
        time.sleep(1)

        self.tracker = ViveTracker(arm)
        # self.tracker.calibrate_pose_zero()


        print("Initializing gripper")
        self.gripper.enable_torque()
        self.gripper.goto_joints([gripper_joints[arm][1]], 2.0)
        time.sleep(2)
        print("Gripper initialized")

        self.reachy_part = arm
        self.init = False
        self.pause = False
        self.mouse = False
        self.top_grasp = {"r_arm" : False, "l_arm" : False}
        self.init_tg = False
        self.position_coeff = 1
        self.orientation_coeff = 1
        self.mobile_base = False
        self.mirror = False

        position = [0.36, -0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        r_pose = self.make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        position = [0.36, 0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        l_pose = self.make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        self.reachy_previous_pose = {"r_arm" : r_pose, "l_arm" : l_pose}
        self.reachy_head_joints = [0, 0, 0]

        self.reachy_init_pose = {"r_arm" : r_pose, "l_arm" : l_pose}

        self.init_reachy()
        self.gripper.disable_torque()

        self.reachy.reset_default_limits()

        self.tracker.update_tracker_pose()

        self.so_init_pose = {"r_arm" : self.tracker.tracker_pose,
                                 "l_arm" : self.tracker.tracker_pose,}
        
        self.so_previous_pose = {"r_arm" : self.rotate_pose(self.tracker.tracker_pose, "r_arm"),
                                 "l_arm" : self.rotate_pose(self.tracker.tracker_pose, "l_arm"),}

        keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        keyboard_thread.start()


    def init_reachy(self):
        self.reachy.turn_on()
        self.reachy.head.l_antenna.turn_on()
        self.reachy.head.r_antenna.turn_on()
        self.reachy.r_arm.gripper.open()
        self.reachy.l_arm.gripper.open()
        joints = self.reachy.r_arm.inverse_kinematics(self.reachy_previous_pose["r_arm"])
        self.reachy.r_arm.goto(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
        joints = self.reachy.l_arm.inverse_kinematics(self.reachy_previous_pose["l_arm"])
        self.reachy.l_arm.goto(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
        self.reachy.head.goto(self.reachy_head_joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)
        # self.init_antenna()

    def make_homogenous_matrix_from_rotation_matrix(self, rotation_matrix, position):
        """Convert a 3x3 rotation matrix to a 4x4 homogenous matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = position
        return matrix
            

    # def init_so(self):
    #     print("init so")
        # print(self.reachy_part)
        # print(self.so_previous_joints[self.reachy_part])
        # if self.reachy_part == "mobile_base":
        #     joints = self.so_init_joints[self.reachy_part]
        # else:
        #     joints =  self.so_previous_joints[self.reachy_part]
        # self.so100.goto_joints(joints, 2.0)

    # def init_top_grasp(self):
    #     self.so100.enable_torque()
    #     print(self.top_grasp[self.reachy_part])
    #     start_orientation = R.from_matrix(self.reachy_previous_pose[self.reachy_part][:3, :3])
    #     # previous_reachy_pose = self.reachy_previous_pose[self.reachy_part]
    #     so_pose = fk(self.so_previous_joints[self.reachy_part], self.top_grasp[self.reachy_part])
    #     pose = self.find_reachy_pose(so_pose, self.reachy_part, self.top_grasp[self.reachy_part])
    #     self.reachy_previous_pose[self.reachy_part] = pose

        # self.go_to_pose(self.reachy, pose, self.reachy_part)
        # time.sleep(2)
        # if self.reachy_part == "r_arm":
        #     self.reachy.r_arm.goto(self.reachy.r_arm.inverse_kinematics(pose), 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)
        # else:
        #     self.reachy.l_arm.goto(self.reachy.l_arm.inverse_kinematics(pose), 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)
        # end_orientation = R.from_matrix(pose[:3, :3])

        # # Définition du nombre de pas d'interpolation
        # frequency = 100  # Nombre de mises à jour par seconde
        # duration = 2.0  # Durée totale de l'interpolation en secondes
        # steps = int(duration * frequency)

        # # Création de l'interpolateur Slerp
        # times = np.linspace(0, 1, steps)  # Valeurs de progression de 0 à 1
        # slerp = Slerp([0, 1], R.concatenate([start_orientation, end_orientation]))

        # # Boucle d'interpolation
        # for i in range(steps):
        #     alpha = i / steps  # Facteur d'interpolation
        #     interpolated_rotation = slerp([alpha])  # Interpolation à l'instant alpha
        #     pose[:3, :3] = interpolated_rotation.as_matrix()[0]  # Convertir en matrice
        #     self.go_to_pose(self.reachy, pose, self.reachy_part)
        #     time.sleep(1 / frequency)
        # self.reachy_previous_pose[self.reachy_part] = pose 
        # self.so100.disable_torque()
        
    # def init_antenna(self):
    #     duration = 3.0
    #     frequency = 100
    #     steps = int(duration * frequency)
    #     r_current_position = self.reachy.head.r_antenna.goal_position
    #     l_current_position = self.reachy.head.l_antenna.goal_position
    #     r_goal_position = -30
    #     l_goal_position = 30
    #     for i in range(steps):
    #         self.reachy.head.r_antenna.goal_position = r_current_position + (r_goal_position - r_current_position) * i / steps
    #         self.reachy.head.l_antenna.goal_position = l_current_position + (l_goal_position - l_current_position) * i / steps
    #         self.reachy.head.send_goal_positions()
    #         time.sleep(1/frequency)

    def go_to_pose(self, reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
        # print("go to pose")
        if arm == "r_arm":
            request = ArmCartesianGoal(
                id=reachy.r_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.CONTINUOUS,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.05),
                order_id=Int32Value(value=5),
            )
            reachy.r_arm._stub.SendArmCartesianGoal(request)
            # print(reachy.r_arm.shoulder.pitch.present_position)
        elif arm == "l_arm":
            request = ArmCartesianGoal(
                id=reachy.l_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.CONTINUOUS,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.05),
                order_id=Int32Value(value=5),
            )
            reachy.l_arm._stub.SendArmCartesianGoal(request)

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


    def find_reachy_pose(self, so_pose, arm):
        reachy_pose = self.reachy_previous_pose[arm].copy()

        diff_position = so_pose[:3, 3] - self.so_previous_pose[arm][:3, 3]
        diff_position *= self.position_coeff

        

        # print(diff_position)

        # diff_position_y = diff_position[1]
        # diff_position[1] = diff_position[2]
        # diff_position[2] = diff_position_y
        # diff_position[0] = -diff_position[0]
        if self.mirror:
            diff_position[1] = -diff_position[1]
        reachy_pose[:3, 3] += diff_position

        diff_orientation = so_pose[:3, :3] @ self.so_previous_pose[arm][:3, :3].T 
        diff_orientation = R.from_matrix(diff_orientation).as_euler('xyz', degrees=False)
        diff_orientation *= self.orientation_coeff
        # print(diff_orientation)
        # print("______________________")
        # # if self.mirror:
        # orientation = diff_orientation[2]
        # diff_orientation[0] = -diff_orientation[0]
        # diff_orientation[2] = diff_orientation[1]
        # diff_orientation[1] = orientation
        diff_orientation = R.from_euler('xyz', diff_orientation, degrees=False).as_matrix()
        reachy_pose[:3, :3] = diff_orientation @ self.reachy_previous_pose[arm][:3, :3]
        # orientation = R.from_euler('xyz', [0, np.pi/2, 0], degrees=False)orientation = T[:3, :3]
        self.so_previous_pose[arm] = so_pose  


        
        return reachy_pose


    def gripper_control(self, joint, arm):
        if arm == "r_arm":
            min_joint, max_joint = -60, -15
        else:
            min_joint, max_joint = 60, 15
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper

        if arm == "r_arm":
            self.reachy.r_arm.gripper.goal_position = gripper_opening
            self.reachy.r_arm.gripper.send_goal_positions()
        elif arm == "l_arm":
            self.reachy.l_arm.gripper.goal_position = gripper_opening
            self.reachy.l_arm.gripper.send_goal_positions()


    def rotate_pose(self, pose, arm):
        
        relative_pose = np.linalg.inv(self.so_init_pose["r_arm"]) @ pose

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

    # def antena_control(self, joint):
    #     min_joint, max_joint = -60, 0
    #     min_antenna, max_antenna = 30, -160
    #     if joint < -60:
    #         joint = -60
    #     if joint > -0:
    #         joint = -0
    #     antenna_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_antenna - min_antenna) + min_antenna

    #     self.reachy.head.r_antenna.goal_position = antenna_opening
    #     self.reachy.head.l_antenna.goal_position = -antenna_opening
    #     self.reachy.send_goal_positions()


    def update_control_arm(self, arm):
        gripper_joints = self.gripper.get_joints()[0]
        pose = self.tracker.tracker_pose

        pose = self.rotate_pose(pose, arm)
        reachy_pose = self.find_reachy_pose(pose, arm)
        # print(reachy_pose)
        self.reachy_previous_pose[arm] = reachy_pose
        self.go_to_pose(self.reachy, reachy_pose, arm)
        self.gripper_control(gripper_joints, arm)

    # def control_head(self, so_pose):
    #     reachy_roll_range = [-40, 40]
    #     reachy_pitch_range = [-40, 40]
    #     so_roll_range = [-100, 0]
    #     so_pitch_range = [-40, 40]

    #     head_roll = np.interp(so_pose[0], so_roll_range, reachy_roll_range)
    #     head_pitch = np.interp(so_pose[1], so_pitch_range, reachy_pitch_range)
    #     head_yaw = so_pose[4] + 65

    #     self.reachy.head.neck.roll.goal_position = head_roll
    #     self.reachy.head.neck.pitch.goal_position = head_pitch
    #     self.reachy.head.neck.yaw.goal_position = head_yaw
    #     self.reachy.head.send_goal_positions()

    #     self.antena_control(so_pose[5])

    #     self.so_previous_pose["head"] = so_pose
    #     self.so_previous_joints["head"] = so_pose



if __name__ == "__main__":
    try:
        # teleop = RobotController("/dev/ttyACM0", "192.168.10.107")
        side = "l_arm"
        teleops = {
            "l_arm" : RobotController("l_arm", "localhost"),
            "r_arm" : RobotController("r_arm", "localhost"),
        }
        # teleop = RobotController(side, "localhost")
        frequency = 100
        time.sleep(1)
        fig, ax = create_plot()

        # while True:
        #     teleop.tracker.update_tracker_pose()
        #     pose = teleop.tracker.tracker_pose
        #     # pose = teleop.rotate_pose(pose)
        #     print(pose[:3, 3])

        #     # orientation = R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True)
        #     # print(f"orientation: {orientation}")

        #     position = pose[:3, 3]
        #     position = position - teleop.so_init_pose["r_arm"][:3, 3]
        #     position = teleop.so_init_pose["r_arm"][:3, :3].T @ position

        #     update_plot(ax, np.array([position]), pose[:3, :3])
        #     time.sleep(0.01)

        #     # teleop.find_reachy_pose2(pose, teleop.reachy_part)
            
        #     # diff_orientation = pose[:3, :3] @ teleop.so_init_pose["r_arm"][:3, :3].T 
        #     # # diff_orientation = R.from_matrix(diff_orientation).as_euler('xyz', degrees=True)
        #     # orientation1 = teleop.so_init_pose[teleop.reachy_part][:3, :3]
        #     # diff_orientation = np.linalg.inv(orientation1) @ diff_orientation
        #     # diff_orientation = R.from_matrix(diff_orientation).as_euler('xyz', degrees=True)
        #     # print(diff_orientation)
        #     # print("______________________")

        while True:
            # print("_______")
            t = time.time()
            # part = teleop.reachy_part

            if teleops["r_arm"].mouse:
                pass
                # joints = teleop.so100.get_joints()
                # pose = teleop.tracker.tracker_pose

                # pose = teleop.find_reachy_pose(fk(joints), part, teleop.top_grasp[part])
                # teleop.so_previous_joints[part] = joints
            elif teleops["r_arm"].init:
                print("switching arm")
                teleops["r_arm"].init_so()
                print("done")
                teleops["r_arm"].init = False
            else :
                # print("ici")
                teleops["r_arm"].tracker.update_tracker_pose()
                teleops["r_arm"].update_control_arm("r_arm")
                teleops["l_arm"].tracker.update_tracker_pose()
                teleops["l_arm"].update_control_arm("l_arm")


            # print(max(0, 1/frequency - (time.time() - t)))
            time.sleep(max(0, 1/frequency - (time.time() - t)))
    except KeyboardInterrupt:
        teleops["r_arm"].so100.close()
        teleops["r_arm"].reachy.turn_off_smoothly()
        print("Exiting...")
        exit(0)
