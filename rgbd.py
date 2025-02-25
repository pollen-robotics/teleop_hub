import time
from collections import deque
from typing import List, Optional

import cv2  # type : ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type : ignore
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from pyorbbecsdk import (  # type : ignore
    Config,
    Frame,
    OBAlignMode,
    OBFormat,
    OBSensorType,
    Pipeline,
)
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

# landmarks constant :
HANDLANDMARKS_CST = [
    4,
    8,
    16,
    20,
]  # Thumb tip, Index finger tip, Ring finger tip, Pinky tip

SHOULDER_CST = [11, 12]
ELBOWS_CST = [13, 14]
WRISTS_CST = [15, 16]

FACELANDMARKS_CST = [1, 152, 33, 263, 61, 291]
# Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner


def rotationMatrixToEulerAngles(rot):
    """
    Convertit une matrice de rotation 3x3 en angles d'Euler (roll, pitch, yaw)
    selon la convention suivante :
      - Roll  : rotation autour de l'axe X
      - Pitch : rotation autour de l'axe Y
      - Yaw   : rotation autour de l'axe Z
    Les angles sont renvoyés en degrés.
    """
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def rotation_matrix_from_vector(vect: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix aligning [0, 0, -1] to the given vect."""
    vect1 = np.array([0, 0, -1])
    eps = 1e-6  # tolérance pour éviter les erreurs numériques

    vect = vect / (np.linalg.norm(vect) + eps)

    # Cas particulier : vecteur aligné ou opposé
    if np.allclose(vect1, vect, atol=eps):
        return np.eye(3)
    if np.allclose(vect1, -vect, atol=eps):
        return np.diag([-1, -1, 1])

    # Calcul de l'axe et de l'angle de rotation
    rotation_vector = np.cross(vect1, vect)
    sin_theta = np.linalg.norm(rotation_vector)
    cos_theta = np.dot(vect1, vect)

    # Construction de la matrice avec l'angle de rotation
    axis = rotation_vector / (sin_theta + eps)
    angle = np.arctan2(sin_theta, cos_theta)
    rotation_matrix = R.from_rotvec(axis * angle).as_matrix()

    return rotation_matrix


class Orbbec:
    def __init__(self, align_mode="SW", enable_sync=True):
        self.config = Config()
        self.pipeline = Pipeline()

        color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = color_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.color_profile)

        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.depth_profile = depth_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.depth_profile)

        self.MIN_DEPTH = 20
        self.MAX_DEPTH = 3500

        self.set_align_mode(align_mode, enable_sync)

        self.pipeline.start(self.config)

        self.color_shape = np.zeros(2, dtype=np.int32)
        self.depth_shape = np.zeros(2, dtype=np.int32)
        self.get_parameters()

    def set_align_mode(self, align_mode, enable_sync):
        device = self.pipeline.get_device()
        device_info = device.get_device_info()
        device_pid = device_info.get_pid()

        if align_mode == "HW":
            if device_pid == 0x066B:
                self.config.set_align_mode(OBAlignMode.SW_MODE)
                print("Mode d'alignement : Software (auto pour Femto Mega)")
            else:
                self.config.set_align_mode(OBAlignMode.HW_MODE)
                print("Mode d'alignement : Hardware")
        elif align_mode == "SW":
            self.config.set_align_mode(OBAlignMode.SW_MODE)
            print("Mode d'alignement : Software")
        else:
            self.config.set_align_mode(OBAlignMode.DISABLE)
            print("Alignement désactivé")

        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
                print("Synchronisation des frames activée")
            except Exception as e:
                print(f"Erreur de synchronisation : {e}")

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Orbbec stopped.")

    def get_parameters(self):
        while True:
            frames = self.pipeline.wait_for_frames(500)
            if frames:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame and depth_frame:
                    self.color_shape[:] = [color_frame.get_height(), color_frame.get_width()]
                    self.depth_shape[:] = [depth_frame.get_height(), depth_frame.get_width()]
                    break

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(500)
        color_frame = frames.get_color_frame() if frames else None
        color_frame_bgr = self.frame_to_bgr_image(color_frame) if color_frame else None
        depth_frame = frames.get_depth_frame() if frames else None
        depth_data = self.get_depth_data(depth_frame)
        return color_frame_bgr, depth_data

    def get_depth_data(self, depth_frame) -> Optional[Frame]:
        if depth_frame:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(self.depth_shape)

            depth_data = depth_data.astype(np.float32)
            depth_data -= self.MIN_DEPTH
            depth_data /= self.MAX_DEPTH - self.MIN_DEPTH
            np.clip(depth_data, 0, 1, out=depth_data)
            depth_data *= 255
            depth_image = depth_data.astype(np.uint8)

            return depth_image
        return None

    def view_stream(self, show_color=True, show_depth=True):
        while True:
            try:
                color_image, depth_image = self.get_frames()

                if show_color and color_image is not None:
                    cv2.imshow("Color Viewer", color_image)

                if show_depth and depth_image is not None:
                    depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                    cv2.imshow("Depth Viewer", depth_colormap)

                key = cv2.waitKey(1)
                if key in [ord("q"), 27]:
                    break
            except KeyboardInterrupt:
                break

    def frame_to_bgr_image(self, frame) -> Optional[np.ndarray]:
        width, height = frame.get_width(), frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())

        if color_format in [OBFormat.RGB, OBFormat.BGR]:
            image = data.reshape((height, width, 3))
            if color_format == OBFormat.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            print(f"Unsupported color format: {color_format}")
            return None
        return image


class MedianFilter:
    def __init__(self, filter_size=5):
        self.filter_size = filter_size
        self.measurements = deque(maxlen=filter_size)

    def update(self, measurement):
        self.measurements.append(measurement)
        if len(self.measurements) == self.filter_size:
            return np.median(np.array(self.measurements), axis=0)
        else:
            return measurement


class KalmanFilter3D:
    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=0.1,
        error_cov_post=1.0,
        add_median_filter=True,
        median_filter_size=5,
    ):
        self.kf = cv2.KalmanFilter(6, 3)

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32  # x  # y  # z
        )

        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 1, 0, 0],  # x = x + vx
                [0, 1, 0, 0, 1, 0],  # y = y + vy
                [0, 0, 1, 0, 0, 1],  # z = z + vz
                [0, 0, 0, 1, 0, 0],  # vx = vx
                [0, 0, 0, 0, 1, 0],  # vy = vy
                [0, 0, 0, 0, 0, 1],  # vz = vz
            ],
            np.float32,
        )

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise

        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise

        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * error_cov_post

        self.initialized = False
        self.kf.statePre = np.zeros((6, 1), dtype=np.float32)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

        self.add_median_filter = add_median_filter
        if self.add_median_filter:
            self.median_filter = MedianFilter(median_filter_size)

    def update(self, measurement):
        if measurement is None:
            return self.kf.statePost[:3].flatten() if self.initialized else None

        measurement = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePre[:3] = measurement
            self.kf.statePost[:3] = measurement
            self.initialized = True
            return measurement.flatten()

        self.kf.correct(measurement)
        prediction = self.kf.predict()
        filtered_position = np.array(prediction[:3]).flatten()

        if self.add_median_filter:
            filtered_position = self.median_filter.update(filtered_position)

        return filtered_position


class ComputerVision:
    def __init__(self, camera: Orbbec):
        # orbbec initialization
        self.camera = camera

        # mediapipe holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )

        # ROI coordinates in image frame
        self.shoulders = np.zeros((2, 3))
        self.elbows = np.zeros((2, 3))
        self.wrists = np.zeros((2, 3))
        self.left_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.face_points = np.zeros((len(FACELANDMARKS_CST), 3))

        self.dist_intershoulder = 0
        self.user_center = np.zeros(3)

        # parameters
        self.image_shape = self.camera.color_shape
        self.camera_matrix = np.eye(3, dtype=np.float32)
        self.calibration_camera()

    def calibration_camera(self):
        focal_length = self.image_shape[1]
        center = (self.image_shape[1] / 2, self.image_shape[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def landmarks_to_xyz(self, landmarks, depth_frame) -> Optional[np.ndarray]:
        if not landmarks:
            return None

        landmarks = np.array([(lm.x, lm.y) for lm in landmarks.landmark])
        x_init = (landmarks[:, 0] * self.image_shape[1]).astype(int)
        y_init = (landmarks[:, 1] * self.image_shape[0]).astype(int)

        x = np.clip(x_init, 0, self.image_shape[1] - 1)
        y = np.clip(y_init, 0, self.image_shape[0] - 1)

        z = depth_frame[y, x]

        return np.vstack((x, y, z)).T.astype(np.float32)

    def get_landmarks_coordinates(self, image, depth_frame, fixed_user=False) -> bool:
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results:
            return False

        body_landmarks = self.landmarks_to_xyz(results.pose_landmarks, depth_frame)
        left_hand_landmarks = self.landmarks_to_xyz(results.left_hand_landmarks, depth_frame)
        right_hand_landmarks = self.landmarks_to_xyz(results.right_hand_landmarks, depth_frame)
        face_landmarks = self.landmarks_to_xyz(results.face_landmarks, depth_frame)

        if body_landmarks is not None:
            if not fixed_user:
                left_shoulder = body_landmarks[SHOULDER_CST[0]]
                right_shoulder = body_landmarks[SHOULDER_CST[1]]
                if self.shoulders[0] is not None and self.shoulders[1] is not None:
                    self.shoulders[0] = left_shoulder
                    self.shoulders[1] = right_shoulder
                    self.user_center = (self.shoulders[0] + self.shoulders[1]) / 2.0
                    self.dist_intershoulder = np.linalg.norm(self.shoulders[0] - self.shoulders[1])

            for i in range(2):
                elbow = body_landmarks[ELBOWS_CST[i]]
                if elbow is not None:
                    self.elbows[i] = elbow
                wrist = body_landmarks[WRISTS_CST[i]]
                if wrist is not None:
                    self.wrists[i] = wrist

        for hand_landmarks, points in zip(
            [left_hand_landmarks, right_hand_landmarks], [self.left_hand_points, self.right_hand_points]
        ):
            if hand_landmarks is not None:
                for i, lm_cst in enumerate(HANDLANDMARKS_CST):
                    finger_coord = hand_landmarks[lm_cst]
                    if finger_coord is not None:
                        points[i] = finger_coord

        if face_landmarks is not None:
            for i, lm_cst in enumerate(FACELANDMARKS_CST):
                face_coord = face_landmarks[lm_cst]
                if face_coord is not None:
                    self.face_points[i] = face_coord

        return True

    def visualization_landmarks(self, color_frame, body_on=True, hands_on=True, face_on=True):
        cv2.circle(color_frame, (int(self.user_center[0]), int(self.user_center[1])), 5, (255, 255, 255), -1)
        if body_on:
            for i in range(2):
                cv2.circle(color_frame, (int(self.wrists[i][0]), int(self.wrists[i][1])), 5, (0, 255, 0), -1)
                cv2.circle(color_frame, (int(self.elbows[i][0]), int(self.elbows[i][1])), 5, (255, 0, 0), -1)
                cv2.circle(color_frame, (int(self.shoulders[i][0]), int(self.shoulders[i][1])), 5, (0, 0, 255), -1)

        if hands_on:
            for i in range(len(HANDLANDMARKS_CST)):
                cv2.circle(
                    color_frame,
                    (int(self.left_hand_points[i][0]), int(self.left_hand_points[i][1])),
                    5,
                    (255, 255, 0),
                    -1,
                )
                cv2.circle(
                    color_frame,
                    (int(self.right_hand_points[i][0]), int(self.right_hand_points[i][1])),
                    5,
                    (0, 255, 255),
                    -1,
                )
        if face_on:
            for i in range(len(FACELANDMARKS_CST)):
                cv2.circle(
                    color_frame,
                    (int(self.face_points[i][0]), int(self.face_points[i][1])),
                    5,
                    (255, 0, 255),
                    -1,
                )

        cv2.imshow("Color Viewer", color_frame)


class RobotController:
    def __init__(self, host):
        self.reachy = ReachySDK(host)
        self.reachy.turn_on()

        # user parameters
        self.real_shoulders = np.array([[0, 0.15, 0], [0, -0.15, 0]])
        self.real_dist_intershoulder = 0.3
        # self.user_center = None

        # control parameters
        self.control_frequency = 100
        self.grippers_ready = [
            self.reachy.l_arm.gripper.is_on(),
            self.reachy.r_arm.gripper.is_on(),
        ]

        # gripper parameters
        self.former_left_hand_points = [deque(maxlen=5) for _ in range(len(HANDLANDMARKS_CST))]
        self.former_right_hand_points = [deque(maxlen=5) for _ in range(len(HANDLANDMARKS_CST))]

        # filters
        self.kf_user_center = KalmanFilter3D()
        self.kf_shoulders = [KalmanFilter3D() for _ in range(2)]
        self.kf_wrists = [KalmanFilter3D() for _ in range(2)]
        self.kf_elbows = [KalmanFilter3D() for _ in range(2)]
        self.kf_left_hand = [KalmanFilter3D(add_median_filter=False) for _ in range(len(HANDLANDMARKS_CST))]
        self.kf_right_hand = [KalmanFilter3D() for _ in range(len(HANDLANDMARKS_CST))]
        self.kf_face = [KalmanFilter3D(median_filter_size=3) for _ in range(len(FACELANDMARKS_CST))]

        self.mf_gripper = MedianFilter()

    # def set_user_center(self, user_center):
    #     self.user_center = user_center

    def convert_to_robot_frame(self, landmark, user_center, dist_intershoulder):
        position_user_frame = landmark - user_center
        normalization_factor = self.real_dist_intershoulder / dist_intershoulder
        x = position_user_frame[0] * normalization_factor
        y = position_user_frame[1] * normalization_factor
        z = position_user_frame[2] * 10e-3
        position_robot_frame = np.array([-z, x, -y])
        return position_robot_frame

    # def is_top_grasp_pose(self, wrist, elbow):
    #     if wrist is None or elbow is None:
    #         return False
    #     return elbow[1] < 0.9 * obj[1]

    def get_effector_pose(self, side, vision):
        side_int = 0 if side == "left" else 1
        wrist_position = self.kf_wrists[side_int].update(vision.wrists[side_int])
        # lbow_position = self.kf_elbows[side_int].update(vision.elbows[side_int])
        user_center = self.kf_user_center.update(vision.user_center)
        left_shoulder = self.kf_shoulders[0].update(vision.shoulders[0])
        right_shoulder = self.kf_shoulders[1].update(vision.shoulders[1])
        dist_intershoulder = np.linalg.norm(left_shoulder - right_shoulder)

        goal_position = self.convert_to_robot_frame(wrist_position, user_center, dist_intershoulder)

        # check if the object is in a top grasp pose
        # if not self.is_top_grasp_pose(obj_center, elbow):
        vect = goal_position - self.real_shoulders[side_int]
        vect = vect / np.linalg.norm(vect)
        rotation_matrix = rotation_matrix_from_vector(vect)

        # else:
        #     print("top grasp")
        #     rot_z = -180 * depth + 90
        #     if side_int == 0:
        #         rot_z = -rot_z
        #     rotation_matrix = R.from_euler("xyz", [0, 0, rot_z], degrees=True).as_matrix()

        goal_pose = recompose_matrix(rotation_matrix, goal_position)
        return goal_pose

    def go_to_pose(self, pose: npt.NDArray[np.float64], arm: str) -> None:
        if arm == "r_arm":
            request = ArmCartesianGoal(
                id=self.reachy.r_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.CONTINUOUS,
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
                continuous_mode=IKContinuousMode.CONTINUOUS,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.05),
                order_id=Int32Value(value=5),
            )
            self.reachy.l_arm._stub.SendArmCartesianGoal(request)

    def make_line(self, end_pose: list[npt.NDArray[np.float64]], duration: float):
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

        nbr_points = int(duration * self.control_frequency)

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

            time.sleep(max(1.0 / self.control_frequency - (time.time() - t), 0.0))

    def get_head_rotation(self, vision):
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Nose tip
                [0.0, -330.0, -65.0],  # Chin
                [-225.0, 170.0, -135.0],  # Left eye left corner
                [225.0, 170.0, -135.0],  # Right eye right corner
                [-150.0, -150.0, -125.0],  # Left mouth corner
                [150.0, -150.0, -125.0],  # Right mouth corner
            ],
            dtype=np.float32,
        )

        dist_coeffs = np.zeros((4, 1))

        face_points_filtered = np.zeros((len(FACELANDMARKS_CST), 3))
        for idx in range(len(FACELANDMARKS_CST)):
            face_points_filtered[idx] = self.kf_face[idx].update(vision.face_points[idx])
        face_points_2D = np.array(face_points_filtered[:, :2], dtype=np.float32)

        success_pnp, rvec, tvec = cv2.solvePnP(
            model_points,
            face_points_2D,
            vision.camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success_pnp:
            rot, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = rotationMatrixToEulerAngles(rot)

            roll_reachy_frame = -yaw
            pitch_reachy_frame = roll - 180
            yaw_reachy_frame = -pitch

        return roll_reachy_frame, pitch_reachy_frame, yaw_reachy_frame

    def set_head_orientation(self, roll, pitch, yaw):
        self.reachy.head.neck.roll.goal_position = roll
        self.reachy.head.neck.pitch.goal_position = pitch
        self.reachy.head.neck.yaw.goal_position = yaw
        self.reachy.send_goal_positions()

    def get_gripper_command(self, side, vision):
        hand_points = vision.left_hand_points if side == "left" else vision.right_hand_points
        kf_hand = self.kf_left_hand if side == "left" else self.kf_right_hand
        gripper = self.reachy.l_arm.gripper if side == "left" else self.reachy.r_arm.gripper
        side_int = 0 if side == "left" else 1

        # get the filtered fingers position
        hand_points_filtered = np.zeros((len(HANDLANDMARKS_CST), 3))
        for idx in range(len(HANDLANDMARKS_CST)):
            hand_points_filtered[idx] = kf_hand[idx].update(hand_points[idx])

        index = hand_points_filtered[1]
        thumb = hand_points_filtered[0]

        dist_index_thumb_normalized = np.linalg.norm(index - thumb) / vision.dist_intershoulder
        dist_filtered = self.mf_gripper.update(dist_index_thumb_normalized)

        if not gripper.is_moving() and dist_filtered < 0.1:
            gripper.close()

        elif not gripper.is_moving() and dist_filtered > 0.3:
            gripper.open()


class TeleopControl:
    def __init__(self, camera: Orbbec, robot_controller: RobotController, timestep: float):
        self.camera = camera
        self.vision = ComputerVision(self.camera)
        self.robot_controller = robot_controller
        self.timestep = timestep
        self.first_pose_done = False

    def run(self):
        # first pose
        while not self.first_pose_done:
            color_frame, depth_frame = self.vision.camera.get_frames()
            if color_frame is not None and depth_frame is not None:
                self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)

                if np.any(self.vision.user_center):
                    left_goal_pose = self.robot_controller.get_effector_pose("left", self.vision)
                    right_goal_pose = self.robot_controller.get_effector_pose("right", self.vision)
                    self.robot_controller.make_line([left_goal_pose, right_goal_pose], 2.0)
                    self.first_pose_done = True
                    print("First pose done")

        # teleoperation
        while True:
            color_frame, depth_frame = self.vision.camera.get_frames()
            if color_frame is not None and depth_frame is not None:
                self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)
                # to visualize landmarks
                # self.vision.visualization_landmarks(color_frame, body_on=False, hands_on=True, face_on=False)

                # make the robot follow the user's hands
                left_goal_pose = self.robot_controller.get_effector_pose("left", self.vision)
                right_goal_pose = self.robot_controller.get_effector_pose("right", self.vision)
                self.robot_controller.make_line([left_goal_pose, right_goal_pose], self.timestep)

                # make the robot follow the user's head
                if np.any(self.vision.face_points):
                    roll, pitch, yaw = self.robot_controller.get_head_rotation(self.vision)
                    self.robot_controller.set_head_orientation(roll, pitch, yaw)

                # make the robot grab the objects
                self.robot_controller.get_gripper_command("left", self.vision)
                self.robot_controller.get_gripper_command("right", self.vision)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        self.camera.stop()


if __name__ == "__main__":
    camera = Orbbec()
    robot = RobotController("localhost")
    teleop = TeleopControl(camera, robot, 0.05)
    teleop.run()
