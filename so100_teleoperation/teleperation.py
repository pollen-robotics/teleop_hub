import numpy as np
import numpy.typing as npt
import time
# import matplotlib
# matplotlib.use("TkAgg")  # Spécifie un backend graphique interactif
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from feetech import Feetech
from reachy2_sdk import ReachySDK

from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
import threading
from pynput import keyboard
from utils import make_homogenous_matrix_from_rotation_matrix, create_plot, fk

SHOW_GRAPH = False


class Teleoperation:

    def __init__(self, port, ip):

        if SHOW_GRAPH:
            fig, ax = create_plot()
        self.so100 = Feetech(port)
        self.reachy = ReachySDK(ip)
        time.sleep(1)
        
        self.reachy_part = "r_arm"
        self.init = False
        self.pause = False
        self.mouse = False
        self.top_grasp = {"r_arm" : False, "l_arm" : False}
        self.init_tg = False
        self.position_coeff = 1
        self.orientation_coeff = 1

        self.so_previous_joints = {"r_arm" : [-58.33, -7.16, 27.74, 72.48, -61.67, -1.8],
                                    "l_arm" : [-58.33, -7.16, 27.74, 72.48, -61.67, -1.8],
                                    "head" : [-50., 0., 27.74, 72.48, -65, -41.05]}
        
        self.so_previous_pose = {"r_arm" : fk(self.so_previous_joints["r_arm"]),
                              "l_arm" : fk(self.so_previous_joints["l_arm"]),
                              "head" : fk(self.so_previous_joints["head"])}
        
        position = [0.36, -0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        r_pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        position = [0.36, 0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        l_pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        self.reachy_previous_pose = {"r_arm" : r_pose, "l_arm" : l_pose}
        self.reachy_head_joints = [0, 0, 0]

        self.so100.enable_torque()
        self.init_so()
        self.init_reachy()
        self.so100.disable_torque()

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
        self.reachy.head.goto(self.reachy_head_joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
        self.init_antenna()
        

    def init_so(self):
        print("init so")
        print(self.reachy_part)
        print(self.so_previous_joints[self.reachy_part])
        self.so100.goto_joints(self.so_previous_joints[self.reachy_part], 2.0)

    def init_top_grasp(self):
        self.so100.enable_torque()
        print(self.top_grasp[self.reachy_part])
        start_orientation = R.from_matrix(self.reachy_previous_pose[self.reachy_part][:3, :3])
        # previous_reachy_pose = self.reachy_previous_pose[self.reachy_part]
        so_pose = fk(self.so_previous_joints[self.reachy_part], self.top_grasp[self.reachy_part])
        pose = self.find_reachy_pose(so_pose, self.reachy_part, self.top_grasp[self.reachy_part])
        self.reachy_previous_pose[self.reachy_part] = pose

        # self.go_to_pose(self.reachy, pose, self.reachy_part)
        # time.sleep(2)
        # if self.reachy_part == "r_arm":
        #     self.reachy.r_arm.goto(self.reachy.r_arm.inverse_kinematics(pose), 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)
        # else:
        #     self.reachy.l_arm.goto(self.reachy.l_arm.inverse_kinematics(pose), 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)
        end_orientation = R.from_matrix(pose[:3, :3])

        # Définition du nombre de pas d'interpolation
        frequency = 100  # Nombre de mises à jour par seconde
        duration = 2.0  # Durée totale de l'interpolation en secondes
        steps = int(duration * frequency)

        # Création de l'interpolateur Slerp
        times = np.linspace(0, 1, steps)  # Valeurs de progression de 0 à 1
        slerp = Slerp([0, 1], R.concatenate([start_orientation, end_orientation]))

        # Boucle d'interpolation
        for i in range(steps):
            alpha = i / steps  # Facteur d'interpolation
            interpolated_rotation = slerp([alpha])  # Interpolation à l'instant alpha
            pose[:3, :3] = interpolated_rotation.as_matrix()[0]  # Convertir en matrice
            self.go_to_pose(self.reachy, pose, self.reachy_part)
            time.sleep(1 / frequency)
        self.reachy_previous_pose[self.reachy_part] = pose 
        self.so100.disable_torque()
        
    def init_antenna(self):
        duration = 2.0
        frequency = 100
        steps = int(duration * frequency)
        r_current_position = self.reachy.head.r_antenna.goal_position
        l_current_position = self.reachy.head.l_antenna.goal_position
        r_goal_position = -30
        l_goal_position = 30
        for i in range(steps):
            self.reachy.head.r_antenna.goal_position = r_current_position + (r_goal_position - r_current_position) * i / steps
            self.reachy.head.l_antenna.goal_position = l_current_position + (l_goal_position - l_current_position) * i / steps
            self.reachy.head.send_goal_positions()
            time.sleep(1/frequency)

    def go_to_pose(self, reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
        # print("go to pose")
        if arm == "r_arm":
            request = ArmCartesianGoal(
                id=reachy.r_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.UNFREEZE,
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
                continuous_mode=IKContinuousMode.UNFREEZE,
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
                if not self.init and not self.pause and self.reachy_part != "l_arm" and not self.mouse:
                    self.init = True
                    self.reachy_part = "l_arm"
                
            elif key.char == 'r':
                if not self.init and not self.pause and self.reachy_part != "r_arm" and not self.mouse:
                    self.init = True
                    self.reachy_part = "r_arm"

            elif key.char == 'p':
                if not self.init and not self.init_tg:
                    self.pause = not self.pause
                    if self.pause:
                        self.so100.enable_torque()
                    else:
                        self.so100.disable_torque()

            elif key.char == 'o':
                if not self.init and not self.pause:
                    self.mouse = not self.mouse

            elif key.char == 't':
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
                    self.init_tg = True
                    self.top_grasp[self.reachy_part] = not self.top_grasp[self.reachy_part]

            elif key.char == 'h':
                if not self.init and not self.pause and not self.init_tg and not self.mouse:
                    self.init = True
                    self.reachy_part = "head"

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

        except AttributeError:
            print(f'Key {key} pressed')

    def listen_keyboard(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()


    def find_reachy_pose(self, so_pose, arm, top_grasp):
        if top_grasp:
            orientation = so_pose[:3, :3]
            if arm == "r_arm":
                rot1 = R.from_euler('xyz', [0, 0, -np.pi/4], degrees=False).as_matrix()
            else:
                rot1 = R.from_euler('xyz', [0, 0, np.pi/4], degrees=False).as_matrix()
            rot2 = R.from_euler('xyz', [0, np.pi/2, 0], degrees=False).as_matrix()
            orientation = orientation @ rot2 @ rot1
            so_pose[:3, :3] = orientation

        reachy_pose = self.reachy_previous_pose[arm].copy()


        # print(reachy_pose)
        diff_position = so_pose[:3, 3] - self.so_previous_pose[arm][:3, 3]
        diff_position *= self.position_coeff
        reachy_pose[:3, 3] += diff_position

        diff_orientation = so_pose[:3, :3] @ self.so_previous_pose[arm][:3, :3].T 
        diff_orientation = R.from_matrix(diff_orientation).as_euler('xyz', degrees=False)
        diff_orientation *= self.orientation_coeff
        diff_orientation = R.from_euler('xyz', diff_orientation, degrees=False).as_matrix()
        reachy_pose[:3, :3] = diff_orientation @ self.reachy_previous_pose[arm][:3, :3]
        # orientation = R.from_euler('xyz', [0, np.pi/2, 0], degrees=False)orientation = T[:3, :3]
        self.so_previous_pose[arm] = so_pose  
        
        return reachy_pose


    def gripper_control(self, joint, arm):
        min_joint, max_joint = -60, -15
        min_gripper, max_gripper = 0, 130
        gripper_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_gripper - min_gripper) + min_gripper

        if arm == "r_arm":
            self.reachy.r_arm.gripper.goal_position = gripper_opening
            self.reachy.r_arm.gripper.send_goal_positions()
        elif arm == "l_arm":
            self.reachy.l_arm.gripper.goal_position = gripper_opening
            self.reachy.l_arm.gripper.send_goal_positions()

    def antena_control(self, joint):
        min_joint, max_joint = -60, 0
        min_antenna, max_antenna = 30, -160
        if joint < -60:
            joint = -60
        if joint > -0:
            joint = -0
        antenna_opening = ((joint - min_joint) / (max_joint - min_joint)) * (max_antenna - min_antenna) + min_antenna

        self.reachy.head.r_antenna.goal_position = antenna_opening
        self.reachy.head.l_antenna.goal_position = -antenna_opening
        self.reachy.send_goal_positions()


    def control_arm(self, so_joints, arm, top_grasp):
        # print(self.top_grasp[arm])
        pose = fk(so_joints, self.top_grasp[arm])
        reachy_pose = self.find_reachy_pose(pose, arm, top_grasp)
        self.reachy_previous_pose[arm] = reachy_pose
        self.go_to_pose(self.reachy, reachy_pose, arm)
        self.gripper_control(so_joints[5], arm)
        self.so_previous_joints[arm] = so_joints

    def control_head(self, so_pose):
        reachy_roll_range = [-40, 40]
        reachy_pitch_range = [-40, 40]
        so_roll_range = [-100, 0]
        so_pitch_range = [-40, 40]

        head_roll = np.interp(so_pose[0], so_roll_range, reachy_roll_range)
        head_pitch = np.interp(so_pose[1], so_pitch_range, reachy_pitch_range)
        head_yaw = so_pose[4] + 65

        self.reachy.head.neck.roll.goal_position = head_roll
        self.reachy.head.neck.pitch.goal_position = head_pitch
        self.reachy.head.neck.yaw.goal_position = head_yaw
        self.reachy.head.send_goal_positions()

        self.antena_control(so_pose[5])

        self.so_previous_pose["head"] = so_pose
        self.so_previous_joints["head"] = so_pose

        


if __name__ == "__main__":
    try:
        teleop = Teleoperation("/dev/ttyACM0", "localhost")
        frequency = 100
        while True:
            # print("_______")
            t = time.time()
            part = teleop.reachy_part
            if part != "head":
                top_grasp = teleop.top_grasp[part]
            if teleop.pause:
                continue
            elif teleop.mouse:
                joints = teleop.so100.get_joints()
                pose = teleop.find_reachy_pose(fk(joints), part, teleop.top_grasp[part])
                teleop.so_previous_joints[part] = joints
            elif teleop.init:
                print("switching arm")
                teleop.init_so()
                print("done")
                teleop.so100.disable_torque()
                teleop.init = False
            elif teleop.init_tg:
                print("init top grasp")
                teleop.init_top_grasp()
                teleop.init_tg = False
            elif part == "head":
                # print("head")
                pose = teleop.so100.get_joints()
                teleop.control_head(pose)
            else :
                # print("ici")
                joints = teleop.so100.get_joints()
                teleop.control_arm(joints, part, top_grasp)
            print(max(0, 1/frequency - (time.time() - t)))
            time.sleep(max(0, 1/frequency - (time.time() - t)))
    except KeyboardInterrupt:
        teleop.so100.close()
        teleop.reachy.turn_off()
        print("Exiting...")
        exit(0)
            

