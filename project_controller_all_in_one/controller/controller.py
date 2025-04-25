import time
from collections import deque

import numpy as np  # type: ignore
import numpy.typing as npt
from controller.filter.kalman_filter import KalmanFilter3D  # type: ignore
from controller.filter.median_filter import MedianFilter  # type: ignore
from controller.filter.rotation_smoother import RotationSmoother  # type: ignore
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from reachy2_sdk import ReachySDK  # type: ignore
from reachy2_sdk.utils.utils import recompose_matrix  # type: ignore
from reachy2_sdk_api.arm_pb2 import (  # type: ignore
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from scipy.spatial.transform import Slerp  # type: ignore
from utils import rotation_matrix_from_vector

SHOULDER_CST = [11, 12]
ELBOWS_CST = [13, 14]
WRISTS_CST = [19, 20]

FACELANDMARKS_CST = [1, 152, 33, 263]  # Nose tip, Chin, Left eye left corner, Right eye right corner
HANDLANDMARKS_CST = [4, 8]  # thumb tip, index tip


class Controller:
    def __init__(self, host):
        self.reachy = ReachySDK(host)
        self.reachy.turn_on()

        # user parameters
        self.real_shoulders = np.array([[0, 0.15, 0], [0, -0.15, 0]])
        self.real_dist_intershoulder = 0.3

        # control parameters
        self.grippers_ready = [
            self.reachy.l_arm.gripper.is_on(),
            self.reachy.r_arm.gripper.is_on(),
        ]

        # filters
        self.kf_user_center = KalmanFilter3D()
        self.kf_shoulders = [KalmanFilter3D() for _ in range(2)]
        self.kf_wrists = [KalmanFilter3D() for _ in range(2)]
        self.kf_elbows = [KalmanFilter3D() for _ in range(2)]
        self.kf_left_hand = [KalmanFilter3D() for _ in range(len(HANDLANDMARKS_CST))]
        self.kf_right_hand = [KalmanFilter3D() for _ in range(len(HANDLANDMARKS_CST))]
        self.kf_face = [KalmanFilter3D(median_filter_size=3) for _ in range(len(FACELANDMARKS_CST))]

        self.mf_gripper = [MedianFilter(), MedianFilter()]
        self.rotation_smoother = [RotationSmoother(), RotationSmoother()]

        self.head_rpy = deque(maxlen=10)

        self.rpy = np.zeros((2, 3))

    def convert_to_robot_frame(self, landmark, user_center, dist_intershoulder):
        position_user_frame = landmark - user_center
        normalization_factor = self.real_dist_intershoulder / dist_intershoulder
        x = position_user_frame[0] * normalization_factor
        y = position_user_frame[1] * normalization_factor
        z = position_user_frame[2] * normalization_factor

        # z = 5 / 3 * z + (0.10)
        z -= 0.15

        z = np.clip(z, -0.7, 0)
        position_robot_frame = np.array([-z, x, -y])
        return position_robot_frame

    def convert_rotation_to_reachy_frame(self, rotation_matrix):
        # conversion dans le repère de Reachy (x = -z, y = x, z = -y)
        T_cam_to_reachy = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        rotation_matrix_reachy = T_cam_to_reachy @ rotation_matrix

        return rotation_matrix_reachy

    def is_top_grasp_pose(self, elbow, user_center, dist_intershoulder):
        if elbow is None:
            return False
        return elbow[1] < user_center[1] + dist_intershoulder / 2

    def rotation_matrix_from_vector(self, side_int, pointA, pointB):
        """Find the rotation matrix that aligns vect1 to vect
        :param vect1: A 3d "source" vector
        :param vect: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vect1, aligns it with vect.
        """
        vect1 = np.array([0, 0, -1])

        vect2 = pointB - pointA
        vect2 = (vect2 / np.linalg.norm(vect2)).reshape(3)

        # handling cross product colinear
        if np.all(np.isclose(vect1, vect2)):
            return np.eye(3)

        # handling cross product colinear
        if np.all(np.isclose(vect1, -vect2)):
            return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        v = np.cross(vect1, vect2)
        c = np.dot(vect1, vect2)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.array(np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2)))

        self.rpy[side_int] = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)

        return rotation_matrix

    def get_effector_pose(self, side, vision, mode="shoulder", dist_filter=True, mirror_mode=False):
        side_int = 0 if side == "left" else 1
        wrist_position = vision.wrists[side_int]
        elbow_position = vision.elbows[side_int]
        user_center = self.kf_user_center.update(vision.user_center)
        shoulders = [
            self.kf_shoulders[0].update(vision.shoulders[0]),
            self.kf_shoulders[1].update(vision.shoulders[1]),
        ]
        dist_intershoulder = np.linalg.norm(shoulders[0] - shoulders[1])

        goal_position = self.convert_to_robot_frame(wrist_position, user_center, dist_intershoulder)
        goal_position_filtered = self.kf_wrists[side_int].update(goal_position)

        if mode == "shoulder":
            # check if the object is in a top grasp pose
            if not self.is_top_grasp_pose(elbow_position, user_center, dist_intershoulder):
                vect = goal_position_filtered - self.real_shoulders[side_int]
                vect = vect / np.linalg.norm(vect)
                rotation_matrix = rotation_matrix_from_vector(vect)

            else:
                rot_z = -5 * goal_position_filtered[0] + 40
                if side_int == 0:
                    rot_z = -rot_z
                rotation_matrix = R.from_euler("xyz", [0, 0, rot_z], degrees=True).as_matrix()

        if mode == "elbow":
            elbow_robot_frame = self.convert_to_robot_frame(elbow_position, user_center, dist_intershoulder)
            elbow_position_filtered = self.kf_elbows[side_int].update(elbow_robot_frame)
            rotation_matrix_wrist = self.rotation_matrix_from_vector(
                side_int, elbow_position_filtered, goal_position_filtered
            )
            rotation_matrix = self.rotation_smoother[side_int].update(rotation_matrix_wrist)

        if mode == "hand":
            rotation_matrix_hand = self.convert_rotation_to_reachy_frame(vision.orientation_hands[side_int])
            rotation_matrix = self.rotation_smoother[side_int].update(rotation_matrix_hand)

        goal_pose = recompose_matrix(rotation_matrix, np.round(goal_position_filtered, 4))

        if self.is_command_unreachable(goal_pose):
            print("Command is unreachable")
            return self.former_poses[side_int]

        if dist_filter and self.is_too_far(goal_pose, side_int):
            print("Goal pose is too far")
            return self.former_poses[side_int]

        self.former_poses[side_int] = goal_pose

        return goal_pose

    def is_too_far(self, goal_pose, side_int):
        goal_position = goal_pose[:3, 3]
        former_position = self.former_poses[side_int][:3, 3]
        return np.linalg.norm(goal_position - former_position) > 0.20

    def is_command_unreachable(self, goal_pose):
        goal_position = goal_pose[:3, 3]
        if (
            goal_position[0] < 0.1
            or goal_position[0] > 0.8
            or goal_position[1] < -0.6
            or goal_position[1] > 0.6
            or goal_position[2] < -0.6
            or goal_position[2] > 0.4
        ):
            print(goal_position)
            return True
        return False

    def go_to_pose(self, pose: npt.NDArray[np.float64], arm: str) -> None:
        if arm == "r_arm":
            request = ArmCartesianGoal(
                id=self.reachy.r_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.UNFREEZE,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.05),
                order_id=Int32Value(value=5),
            )
            self.reachy.r_arm._stub.SendArmCartesianGoal(request)

        elif arm == "l_arm":
            request = ArmCartesianGoal(
                id=self.reachy.l_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.UNFREEZE,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.05),
                order_id=Int32Value(value=5),
            )
            self.reachy.l_arm._stub.SendArmCartesianGoal(request)

    def make_line(self, end_pose: list[npt.NDArray[np.float64]], duration: float):
        control_frequency = 100
        start_pose = [
            self.reachy.l_arm.forward_kinematics(),
            self.reachy.r_arm.forward_kinematics(),
        ]
        start_position = [start_pose[0][:3, 3], start_pose[1][:3, 3]]
        end_position = [end_pose[0][:3, 3], end_pose[1][:3, 3]]
        start_rotation = [
            R.from_matrix(start_pose[0][:3, :3]),
            R.from_matrix(start_pose[1][:3, :3]),
        ]
        end_rotation = [
            R.from_matrix(end_pose[0][:3, :3]),
            R.from_matrix(end_pose[1][:3, :3]),
        ]

        nbr_points = int(duration * control_frequency)

        left_slerp = Slerp(
            [0, 1],
            R.from_matrix([start_rotation[0].as_matrix(), end_rotation[0].as_matrix()]),
        )
        right_slerp = Slerp(
            [0, 1],
            R.from_matrix([start_rotation[1].as_matrix(), end_rotation[1].as_matrix()]),
        )

        for i in range(nbr_points):
            t = time.time()
            alpha = i / nbr_points

            left_interp_rotation = left_slerp([alpha]).as_matrix()[0]
            left_interp_position = start_position[0] + alpha * (end_position[0] - start_position[0])

            right_interp_rotation = right_slerp([alpha]).as_matrix()[0]
            right_interp_position = start_position[1] + alpha * (end_position[1] - start_position[1])

            left_pose = recompose_matrix(left_interp_rotation, left_interp_position)
            right_pose = recompose_matrix(right_interp_rotation, right_interp_position)

            self.go_to_pose(left_pose, "l_arm")
            self.go_to_pose(right_pose, "r_arm")
            time.sleep(max(1.0 / control_frequency - (time.time() - t), 0.0))

    def get_head_rotation(self, vision, mirror_mode=False):
        face_points_filtered = np.zeros((len(FACELANDMARKS_CST), 3))
        for idx in range(len(FACELANDMARKS_CST)):
            face_point_reachy_frame = self.convert_to_robot_frame(
                vision.face_points[idx], vision.user_center, vision.dist_intershoulder
            )
            face_point_filtered = self.kf_face[idx].update(face_point_reachy_frame)
            face_points_filtered[idx] = face_point_filtered

        left_eye = face_points_filtered[2]
        right_eye = face_points_filtered[3]
        nose = face_points_filtered[0]
        chin = face_points_filtered[1]

        y_axis = right_eye - left_eye
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = nose - chin
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        rot = np.array([x_axis, y_axis, z_axis]).T
        roll, pitch, yaw = R.from_matrix(rot).as_euler("XYZ", degrees=True)

        for angle in [roll, pitch, yaw]:
            if angle > 180:
                angle = angle - 360
            elif angle < -180:
                angle = angle + 360

        if mirror_mode:
            return -roll, pitch, -yaw

        return roll, pitch, yaw

    def set_head_orientation(self, roll, pitch, yaw):
        self.head_rpy.append([roll, pitch, yaw])
        self.reachy.head.neck.roll.goal_position = roll
        self.reachy.head.neck.pitch.goal_position = pitch
        self.reachy.head.neck.yaw.goal_position = yaw
        self.reachy.send_goal_positions(check_positions=False)

    def get_gripper_command(self, side, vision, mirror_mode=False):
        if side == "left":
            hand_points = vision.left_hand_points
            kf_hand = self.kf_left_hand
            if not mirror_mode:
                gripper = self.reachy.l_arm.gripper
            else:
                gripper = self.reachy.r_arm.gripper
            side_int = 0
        else:
            hand_points = vision.right_hand_points
            kf_hand = self.kf_right_hand
            if not mirror_mode:
                gripper = self.reachy.r_arm.gripper
            else:
                gripper = self.reachy.l_arm.gripper
            side_int = 1

        # get the filtered fingers position
        hand_points_filtered = np.zeros((len(HANDLANDMARKS_CST), 3))
        for idx in range(len(HANDLANDMARKS_CST)):
            hand_points_filtered[idx] = kf_hand[idx].update(hand_points[idx])

        index = hand_points_filtered[1]
        thumb = hand_points_filtered[0]

        dist_index_thumb_normalized = np.linalg.norm(index - thumb) / vision.dist_intershoulder
        dist_filtered = self.mf_gripper[side_int].update(dist_index_thumb_normalized)
        if not gripper.is_moving() and dist_filtered < 0.1:
            gripper.close()

        elif not gripper.is_moving() and dist_filtered > 0.3:
            gripper.open()

    def get_mirror_pose(self, pose):
        new_pose = np.copy(pose)
        new_pose[1, 3] = -new_pose[1, 3]
        rotation_matrix = new_pose[:3, :3]
        rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()

        mirrored_quaternion = np.array(
            [-rotation_quaternion[0], rotation_quaternion[1], -rotation_quaternion[2], rotation_quaternion[3]]
        )

        mirrored_rotation_matrix = R.from_quat(mirrored_quaternion).as_matrix()
        new_pose[:3, :3] = mirrored_rotation_matrix
        return new_pose
