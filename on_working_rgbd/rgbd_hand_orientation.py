import time
from collections import deque
from typing import List, Optional

import cv2  # type : ignore
import matplotlib.pyplot as plt
import mediapipe as mp  # type: ignore
import numpy as np  # type : ignore
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from mpl_toolkits.mplot3d import Axes3D
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

# original_landmarks constant :
HANDLANDMARKS_CST = [0, 2, 4, 5, 8, 9, 12, 17, 20]
# wrist, thumb mcp, thumb tip, index mcp, index tip, middle mcp, middletip, pinky mcp, pinky tip

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
        self.MAX_DEPTH = 4000

        self.set_align_mode(align_mode, enable_sync)

        self.pipeline.start(self.config)

        self.color_shape = np.zeros(2, dtype=np.int32)
        self.depth_shape = np.zeros(2, dtype=np.int32)

        self.color_format = None
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
                    self.color_format = color_frame.get_format()
                    break

    def get_frames(self, scale_percent=100):
        frames = self.pipeline.wait_for_frames(500)
        if frames:
            color_frame = frames.get_color_frame()
            color_frame_bgr = self.frame_to_bgr_image(color_frame) if color_frame else None
            color_frame_resized = (
                self.resize_frames(color_frame_bgr, scale_percent) if color_frame_bgr is not None else None
            )

            depth_frame = frames.get_depth_frame()
            depth_data = self.get_depth_data(depth_frame)
            depth_data_resized = self.resize_frames(depth_data, scale_percent) if depth_data is not None else None

            return color_frame_resized, depth_data_resized

        return None, None

    def resize_frames(self, frame, scale_percent=50):
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        return frame_resized

    def get_depth_data(self, depth_frame) -> Optional[Frame]:
        if depth_frame:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(self.depth_shape)
            depth_data = depth_data.astype(np.float32) * 10e-4
            return depth_data
        return None

    def view_stream(self, show_color=True, show_depth=True):
        while True:
            try:
                color_data, depth_data = self.get_frames()

                if show_color and color_data is not None:
                    cv2.imshow("Color Viewer", color_data)

                if show_depth and depth_data is not None:
                    depth_image = np.clip(
                        (depth_data - self.MIN_DEPTH) / (self.MAX_DEPTH - self.MIN_DEPTH) * 255, 0, 255
                    )
                    depth_colormap = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imshow("Depth Viewer", depth_colormap)

                key = cv2.waitKey(1)
                if key in [ord("q"), 27]:
                    break
            except KeyboardInterrupt:
                break

    def frame_to_bgr_image(self, frame) -> Optional[np.ndarray]:
        data = np.asanyarray(frame.get_data())

        if self.color_format in [OBFormat.RGB, OBFormat.BGR]:
            image = data.reshape((self.color_shape[0], self.color_shape[1], 3))
            if self.color_format == OBFormat.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif self.color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            print(f"Unsupported color format: {self.color_format}")
            return None
        return image


class RotationSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, new_rotation_matrix):
        if new_rotation_matrix is not None:
            self.buffer.append(new_rotation_matrix)
        return np.mean(self.buffer, axis=0) if self.buffer else new_rotation_matrix


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
    def __init__(self, camera: Orbbec, scale_percent: int = 100):
        # orbbec initialization
        self.camera = camera
        self.scale_percent = scale_percent

        # mediapipe holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
        )

        # ROI coordinates in image frame
        self.shoulders = np.zeros((2, 3))
        self.elbows = np.zeros((2, 3))
        self.wrists = np.zeros((2, 3))
        self.left_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.face_points = np.zeros((len(FACELANDMARKS_CST), 3))
        self.left_hand_points_3D = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points_3D = np.zeros((len(HANDLANDMARKS_CST), 3))

        self.kf_left_hand = [KalmanFilter3D(add_median_filter=False) for _ in range(len(HANDLANDMARKS_CST))]
        self.kf_right_hand = [KalmanFilter3D(add_median_filter=False) for _ in range(len(HANDLANDMARKS_CST))]

        self.orientation_hands = [np.eye(3) for _ in range(2)]
        self.dist_intershoulder = 0
        self.user_center = np.zeros(3)

        # parameters
        self.image_shape = self.camera.color_shape * self.scale_percent // 100
        print("Image shape : ", self.image_shape)
        self.camera_matrix = np.eye(3, dtype=np.float32)
        self.calibration_camera()

    def calibration_camera(self):
        focal_length = self.image_shape[1]
        center = (self.image_shape[1] / 2, self.image_shape[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def get_2D_landmarks(self, results):
        landmarks_dict = {
            "pose": None,
            "face": None,
            "left_hand": None,
            "right_hand": None,
        }

        landmarks_sources = {
            "pose": results.pose_landmarks,
            "face": results.face_landmarks,
            "left_hand": results.left_hand_landmarks,
            "right_hand": results.right_hand_landmarks,
        }

        for key, original_landmarks in landmarks_sources.items():
            if original_landmarks is None:
                continue

            original_landmarks = np.array([(lm.x, lm.y) for lm in original_landmarks.landmark])
            x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
            y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)

            x = np.clip(x_init, 0, self.image_shape[1] - 1)
            y = np.clip(y_init, 0, self.image_shape[0] - 1)
            landmarks_dict[key] = np.vstack((x, y)).T.astype(np.float32)

        return (
            landmarks_dict["pose"],
            landmarks_dict["face"],
            landmarks_dict["left_hand"],
            landmarks_dict["right_hand"],
        )

    def get_3D_landmarks(self, results):
        hands = [
            ("left", results.left_hand_landmarks),
            ("right", results.right_hand_landmarks),
        ]

        landmarks_dict = {"left": None, "right": None}

        for hand, original_landmarks in hands:
            if original_landmarks is None:
                continue

            original_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in original_landmarks.landmark])
            x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
            y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)
            z = (original_landmarks[:, 2] * self.image_shape[1]).astype(int)

            x = np.clip(x_init, 0, self.image_shape[1] - 1)
            y = np.clip(y_init, 0, self.image_shape[0] - 1)
            landmarks_dict[hand] = np.vstack((x, y, z)).T.astype(np.float32)

        return [landmarks_dict["left"], landmarks_dict["right"]]

    def get_hand_orientation(self, handlandmarks_both_side):
        for side_int in range(2):
            kf_hand = self.kf_left_hand if side_int == 0 else self.kf_right_hand
            handpoints = self.left_hand_points_3D if side_int == 0 else self.right_hand_points_3D
            handlandmarks = handlandmarks_both_side[side_int]
            if handlandmarks is not None:
                for ite, lm_cst in enumerate(HANDLANDMARKS_CST):
                    finger_coord = handlandmarks[lm_cst]
                    if finger_coord is not None:
                        coord_filtered = kf_hand[ite].update(finger_coord)
                        handpoints[ite] = coord_filtered
                if side_int == 0:
                    self.left_hand_points_3D = handpoints
                else:
                    self.right_hand_points_3D = handpoints

            wrist = handpoints[0]
            thumb_mcp = handpoints[1]
            index_mcp = handpoints[3]
            thumb_tip = handpoints[2]
            index_tip = handpoints[4]
            mid_index_thumb = (thumb_tip + index_tip) / 2
            # middle_base = np.array([-handpoints[4, 2], handpoints[4, 0], -handpoints[4, 1]])
            # middle_tip = np.array([-handpoints[5, 2], handpoints[5, 0], -handpoints[5, 1]])

            if side_int == 1:
                x_axis = thumb_mcp - index_mcp
            else:
                x_axis = index_mcp - thumb_mcp
                # x_axis = thumb_mcp - index_mcp
            x_axis = normalize_vector(x_axis)
            z_axis = wrist - mid_index_thumb
            z_axis = normalize_vector(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = normalize_vector(y_axis)

            # Vérifier l'orthogonalité et réajuster
            x_axis = np.cross(y_axis, z_axis)

            self.orientation_hands[side_int] = np.column_stack((x_axis, y_axis, z_axis))

    def estimate_3D_landmark_with_depth(self, landmark, depth_frame, radius=2) -> Optional[np.ndarray]:
        if landmark is None:
            return None

        x, y = landmark[0], landmark[1]
        if (radius + 1 < x < self.image_shape[1] - radius - 1) and (radius + 1 < y < self.image_shape[0] - radius - 1):
            z = np.median(depth_frame[int(y - radius) : int(y + radius), int(x - radius) : int(x + radius)])
        else:
            z = depth_frame[int(y), int(x)]

        return np.vstack((x, y, z)).T.astype(np.float32)

    def get_landmarks_coordinates(self, image, depth_frame, fixed_user=False) -> bool:
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results:
            return False

        body_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks = self.get_2D_landmarks(results)
        handpoints_3D = self.get_3D_landmarks(results)
        self.get_hand_orientation(handpoints_3D)

        if body_landmarks is not None:
            if not fixed_user:
                left_shoulder = self.estimate_3D_landmark_with_depth(body_landmarks[SHOULDER_CST[0]], depth_frame)
                if left_shoulder is not None:
                    self.shoulders[0] = left_shoulder
                right_shoulder = self.estimate_3D_landmark_with_depth(body_landmarks[SHOULDER_CST[1]], depth_frame)
                if right_shoulder is not None:
                    self.shoulders[1] = right_shoulder
                if np.any(self.shoulders[0]) and np.any(self.shoulders[1]):
                    self.user_center = (self.shoulders[0] + self.shoulders[1]) / 2.0
                    self.dist_intershoulder = np.linalg.norm(self.shoulders[0] - self.shoulders[1])

            for i in range(2):
                elbow = self.estimate_3D_landmark_with_depth(body_landmarks[ELBOWS_CST[i]], depth_frame)
                if elbow is not None:
                    self.elbows[i] = elbow
                wrist = self.estimate_3D_landmark_with_depth(body_landmarks[WRISTS_CST[i]], depth_frame)
                if wrist is not None:
                    self.wrists[i] = wrist

        for hand_landmarks, points in zip(
            [left_hand_landmarks, right_hand_landmarks], [self.left_hand_points, self.right_hand_points]
        ):
            if hand_landmarks is not None:
                for i, lm_cst in enumerate(HANDLANDMARKS_CST):
                    finger_coord = self.estimate_3D_landmark_with_depth(hand_landmarks[lm_cst], depth_frame)
                    if finger_coord is not None:
                        points[i] = finger_coord

        if face_landmarks is not None:
            for i, lm_cst in enumerate(FACELANDMARKS_CST):
                face_coord = self.estimate_3D_landmark_with_depth(face_landmarks[lm_cst], depth_frame)
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

    def init_plot(self):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.scatter(0, 0, 0, c="r", marker="o")
        fig.canvas.draw()
        return fig, ax

    def update_plot(self, fig, ax):
        ax.clear()
        for point in self.left_hand_points_3D:
            ax.scatter(point[0], point[1], point[2], c="b", marker="o")
        fig.canvas.draw()


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
        self.rotation_smoother = [RotationSmoother(window_size=5), RotationSmoother(window_size=5)]

    # def set_user_center(self, user_center):
    #     self.user_center = user_center

    def convert_to_robot_frame(self, landmark, user_center, dist_intershoulder):
        position_user_frame = landmark - user_center
        normalization_factor = self.real_dist_intershoulder / dist_intershoulder
        x = position_user_frame[0] * normalization_factor
        y = position_user_frame[1] * normalization_factor
        z = position_user_frame[2]
        z = np.clip(z, -0.8, 0)
        position_robot_frame = np.array([-z, x, -y])
        return position_robot_frame

    def is_top_grasp_pose(self, elbow, user_center, dist_intershoulder):
        if elbow is None:
            return False
        return elbow[1] < user_center[1] + dist_intershoulder / 2

    def convert_rotation_to_reachy_frame(self, rotation_matrix):
        # conversion dans le repère de Reachy (x = -z, y = x, z = -y)
        T_cam_to_reachy = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

        # Appliquer la correction après transformation
        rotation_matrix_reachy = T_cam_to_reachy @ rotation_matrix

        return rotation_matrix_reachy

    # def get_forearm_orientation(self, elbow, wrist, reference_direction=np.array([0, -1, 0])):
    #     # Axe Z (Direction du coude vers le poignet)
    #     z_axis = elbow - wrist
    #     z_axis = normalize_vector(z_axis)

    #     # Axe X (stabilisé en utilisant une direction de référence)
    #     x_axis = np.cross(z_axis, reference_direction)  # Perpendiculaire à Z et à l’axe de référence
    #     x_axis = normalize_vector(x_axis)

    #     # Axe Y (perpendiculaire aux deux autres)
    #     y_axis = np.cross(z_axis, x_axis)
    #     y_axis = normalize_vector(y_axis)

    #     # Vérification de l'orthogonalité (corrige les erreurs numériques)
    #     x_axis = np.cross(y_axis, z_axis)

    #     # Matrice de rotation de l'avant-bras
    #     rotation_matrix_forearm = np.column_stack((x_axis, y_axis, z_axis))

    def get_forearm_orientation(self, elbow, wrist):
        """
        Calcule la rotation minimale pour aligner un vecteur de référence avec le vecteur coude → poignet.

        - elbow : np.array([x, y, z]) -> Position du coude dans le repère du corps.
        - wrist : np.array([x, y, z]) -> Position du poignet dans le repère du corps.

        Retourne :
        - rotation_matrix : Matrice de rotation 3x3
        - quaternion : Quaternion (x, y, z, w)
        """

        # Vecteur cible (direction coude → poignet dans le repère du corps)
        forearm_vector = wrist - elbow
        forearm_vector /= np.linalg.norm(forearm_vector)  # Normalisation

        # Vecteur de référence (ex: l'axe Z du corps [0, 0, 1])
        reference_vector = np.array([0, 0, 1])

        # Calcul de la rotation qui aligne `reference_vector` avec `forearm_vector`
        rotation, _ = R.align_vectors([forearm_vector], [reference_vector])

        # Matrice de rotation
        rotation_matrix = rotation.as_matrix()

        return rotation_matrix

    def estimate_elbow_depth(self, shoulder_3D, wrist_3D, elbow):
        x_e, y_e, z_e = shoulder_3D[0], shoulder_3D[1], shoulder_3D[2]
        y_c, z_c = elbow[1], elbow[2]  # Coordonnées 2D détectées du coude

        # Calculer la profondeur z_c en respectant la sphère de rayon L_bras
        delta_z = z_c - z_e
        delta_y = y_c - y_e
        squared_term = (0.35) ** 2 - delta_z**2 - delta_y**2

        if squared_term < 0:
            print("Erreur : Les coordonnées 2D du coude sont incohérentes avec la longueur du bras")
            return elbow

        x_c1 = x_e + np.sqrt(squared_term)
        x_c2 = x_e - np.sqrt(squared_term)

        # Sélectionner la solution la plus proche du poignet
        x_w = wrist_3D[2]
        x_c = x_c1 if abs(x_c1 - x_w) < abs(x_c2 - x_w) else x_c2

        return np.array([x_c, y_c, z_c])

    def get_effector_pose(self, side, vision):
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

        # version avec épaule
        # check if the object is in a top grasp pose
        # if not self.is_top_grasp_pose(elbow_position, user_center, dist_intershoulder):
        # vect = goal_position_filtered - self.real_shoulders[side_int]
        # vect = vect / np.linalg.norm(vect)
        # rotation_matrix_wrist = rotation_matrix_from_vector(vect)

        # else:
        #     rot_z = -5 * goal_position_filtered[0] + 40
        #     if side_int == 0:
        #         rot_z = -rot_z
        #     rotation_matrix_wrist = R.from_euler("xyz", [0, 0, rot_z], degrees=True).as_matrix()

        # version avec coude
        elbow_robot_frame = self.convert_to_robot_frame(elbow_position, user_center, dist_intershoulder)
        elbow_position_filtered = self.kf_elbows[side_int].update(elbow_robot_frame)
        # elbow_new_depth = self.estimate_elbow_depth(
        #     self.real_shoulders[side_int], goal_position_filtered, elbow_position_filtered
        # )
        print("elbow", elbow_position_filtered)
        print("goal_position_filtered", goal_position_filtered)
        rotation_matrix = self.get_forearm_orientation(elbow_position_filtered, goal_position_filtered)

        rotation_matrix_hand = self.convert_rotation_to_reachy_frame(vision.orientation_hands[side_int])
        # rotation_matrix = rotation_matrix_wrist @ rotation_matrix_hand
        rotation_matrix = self.rotation_smoother[side_int].update(rotation_matrix_hand)

        goal_pose = recompose_matrix(rotation_matrix, np.round(goal_position_filtered, 3))
        return goal_pose

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
        self.reachy.send_goal_positions(check_positions=False)

    def get_gripper_command(self, side, vision):
        hand_points = vision.left_hand_points if side == "left" else vision.right_hand_points
        gripper = self.reachy.l_arm.gripper if side == "left" else self.reachy.r_arm.gripper

        index = hand_points[4]
        thumb = hand_points[2]

        dist_index_thumb_normalized = np.linalg.norm(index - thumb) / vision.dist_intershoulder
        dist_filtered = self.mf_gripper.update(dist_index_thumb_normalized)

        if not gripper.is_moving() and dist_filtered < 0.1:
            gripper.close()

        elif not gripper.is_moving() and dist_filtered > 0.3:
            gripper.open()


class TeleopControl:
    def __init__(self, camera: Orbbec, robot_controller: RobotController, scale_percent: int, timestep: float):
        self.camera = camera
        self.vision = ComputerVision(self.camera, scale_percent)
        self.robot_controller = robot_controller
        self.timestep = timestep
        self.first_pose_done = False

    def run(self):
        fig, ax = self.vision.init_plot()

        # # first pose
        # while not self.first_pose_done:
        #     color_frame, depth_frame = self.vision.camera.get_frames(scale_percent=50)
        #     if color_frame is not None and depth_frame is not None:
        #         self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)
        #         if (
        #             np.any(self.vision.user_center)
        #             and np.any(self.vision.wrists)
        #             and np.any(self.vision.left_hand_points)
        #             and np.any(self.vision.right_hand_points)
        #         ):
        #             left_goal_pose = self.robot_controller.get_effector_pose("left", self.vision)
        #             right_goal_pose = self.robot_controller.get_effector_pose("right", self.vision)
        #             self.robot_controller.make_line([left_goal_pose, right_goal_pose], 2.0)
        #             self.first_pose_done = True
        #             print("First pose done")

        # teleoperation
        while True:
            color_frame, depth_frame = self.vision.camera.get_frames(scale_percent=50)
            if color_frame is not None and depth_frame is not None:
                self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)  # too long (0.035)
                self.vision.update_plot(fig, ax)

                # # make the robot follow the user's hands
                # left_goal_pose = self.robot_controller.get_effector_pose("left", self.vision)
                # right_goal_pose = self.robot_controller.get_effector_pose("right", self.vision)
                # if np.any(left_goal_pose) and np.any(right_goal_pose):
                #     left_rpy = R.from_matrix(left_goal_pose[:3, :3]).as_euler("xyz", degrees=True)
                #     right_rpy = R.from_matrix(right_goal_pose[:3, :3]).as_euler("xyz", degrees=True)
                #     cv2.putText(
                #         color_frame,
                #         f"left_goal: {left_rpy}",
                #         (10, 30),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         1,
                #         (255, 255, 255),
                #         2,
                #         cv2.LINE_AA,
                #     )
                #     cv2.putText(
                #         color_frame,
                #         f"right_goal: {right_rpy}",
                #         (10, 60),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         1,
                #         (255, 255, 255),
                #         2,
                #         cv2.LINE_AA,
                #     )
                # # self.robot_controller.make_line([left_goal_pose, right_goal_pose], self.timestep)  # too long (0.035)
                # self.robot_controller.go_to_pose(left_goal_pose, "l_arm")
                # self.robot_controller.go_to_pose(right_goal_pose, "r_arm")

                # # make the robot follow the user's head
                # if np.any(self.vision.face_points):
                #     roll, pitch, yaw = self.robot_controller.get_head_rotation(self.vision)
                #     self.robot_controller.set_head_orientation(roll, pitch, yaw)

                # # make the robot grab the objects
                # self.robot_controller.get_gripper_command("left", self.vision)
                # self.robot_controller.get_gripper_command("right", self.vision)

                # # to visualize original_landmarks
                self.vision.visualization_landmarks(color_frame, body_on=True, hands_on=True, face_on=False)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        self.camera.stop()


if __name__ == "__main__":
    camera = Orbbec()
    scale_percent = 50
    timestep = 0.02
    robot = RobotController("localhost")
    teleop = TeleopControl(camera, robot, scale_percent, timestep)
    teleop.run()
