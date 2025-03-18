import collections
import threading
import time
from collections import deque
from typing import Optional

import cv2  # type : ignore
import matplotlib.pyplot as plt  # type : ignore
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

SHOULDER_CST = [11, 12]
ELBOWS_CST = [13, 14]
# WRISTS_CST = [15, 16]
# INDEX_MCP_CST = [19, 20]
WRISTS_CST = [19, 20]

FACELANDMARKS_CST = [1, 152, 33, 263]  # Nose tip, Chin, Left eye left corner, Right eye right corner
HANDLANDMARKS_CST = [4, 8]  # thumb tip, index tip


def normalize_vector(v):
    """Normalise un vecteur et gère le cas où la norme est zéro."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


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

        self.color_frame = collections.deque(maxlen=1)
        self.depth_frame = collections.deque(maxlen=1)

        self.color_shape = np.zeros(2, dtype=np.int32)
        self.depth_shape = np.zeros(2, dtype=np.int32)

        self.color_format = None

        self.get_parameters()

        self.frame_getter = threading.Thread(target=self.get_frames)
        self.frame_getter.start()

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

    def get_frames(self):
        while True:
            frames = self.pipeline.wait_for_frames(500)
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_frame_bgr = self.frame_to_bgr_image(color_frame)
                    self.color_frame.append(color_frame_bgr)

                depth_frame = frames.get_depth_frame()
                depth_data = self.get_depth_data(depth_frame)
                if depth_data is not None:
                    self.depth_frame.append(depth_data)

    def get_depth_data(self, depth_frame) -> Optional[Frame]:
        if depth_frame and np.any(self.depth_shape):
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(self.depth_shape)
            depth_data = depth_data.astype(np.float32) * 10e-4
            return depth_data
        return None

    def view_stream(self, show_color=True, show_depth=True):
        while True:
            try:
                color_data = self.color_frame
                depth_data = self.depth_frame

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
    # def __init__(self, window_size=8):
    #     self.window_size = window_size
    #     self.buffer = deque(maxlen=window_size)

    # def update(self, new_rotation_matrix):
    #     if new_rotation_matrix is not None:
    #         self.buffer.append(new_rotation_matrix)
    #     return np.median(self.buffer, axis=0) if self.buffer else new_rotation_matrix

    def __init__(self, alpha=0.3):
        """
        Initialise un filtre basé sur une moyenne exponentielle des matrices de rotation.

        - alpha : Coefficient de lissage entre 0 (très lisse) et 1 (aucun filtrage).
        """
        self.alpha = alpha
        self.last_rotation_matrix = None  # Stocke la dernière rotation lissée

    def update(self, current_rotation_matrix):
        """
        Applique un lissage EMA sur la rotation.

        - current_rotation_matrix : Matrice 3x3 représentant l'orientation actuelle.

        Retourne :
        - Matrice de rotation lissée 3x3.
        """
        if self.last_rotation_matrix is None:
            self.last_rotation_matrix = current_rotation_matrix
            return current_rotation_matrix  # Retourne directement la première valeur

        # Moyenne exponentielle sur les composantes de la matrice de rotation
        smoothed_rotation_matrix = (1 - self.alpha) * self.last_rotation_matrix + self.alpha * current_rotation_matrix

        # Réorthogonalisation pour rester une vraie matrice de rotation
        U, _, Vt = np.linalg.svd(smoothed_rotation_matrix)
        smoothed_rotation_matrix = U @ Vt

        # Mise à jour
        self.last_rotation_matrix = smoothed_rotation_matrix
        return smoothed_rotation_matrix


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
    # def __init__(
    #     self,
    #     process_noise=0.001,
    #     measurement_noise=0.1,
    #     error_cov_post=1.0,
    #     add_median_filter=True,
    #     median_filter_size=5,
    # ):
    #     self.kf = cv2.KalmanFilter(6, 3)

    #     self.kf.measurementMatrix = np.array(
    #         [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32  # x  # y  # z
    #     )

    #     self.kf.transitionMatrix = np.array(
    #         [
    #             [1, 0, 0, 1, 0, 0],  # x = x + vx
    #             [0, 1, 0, 0, 1, 0],  # y = y + vy
    #             [0, 0, 1, 0, 0, 1],  # z = z + vz
    #             [0, 0, 0, 1, 0, 0],  # vx = vx
    #             [0, 0, 0, 0, 1, 0],  # vy = vy
    #             [0, 0, 0, 0, 0, 1],  # vz = vz
    #         ],
    #         np.float32,
    #     )

    #     self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise

    #     self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise

    #     self.kf.errorCovPost = np.eye(6, dtype=np.float32) * error_cov_post

    #     self.initialized = False
    #     self.kf.statePre = np.zeros((6, 1), dtype=np.float32)
    #     self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    #     self.add_median_filter = add_median_filter
    #     if self.add_median_filter:
    #         self.median_filter = MedianFilter(median_filter_size)

    # def update(self, measurement):
    #     if measurement is None:
    #         return self.kf.statePost[:3].flatten() if self.initialized else None

    #     measurement = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)

    #     if not self.initialized:
    #         self.kf.statePre[:3] = measurement
    #         self.kf.statePost[:3] = measurement
    #         self.initialized = True
    #         return measurement.flatten()

    #     self.kf.correct(measurement)
    #     prediction = self.kf.predict()
    #     filtered_position = np.array(prediction[:3]).flatten()

    #     if self.add_median_filter:
    #         filtered_position = self.median_filter.update(filtered_position)

    #     return filtered_position

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=0.1,
        error_cov_post=1.0,
        add_median_filter=True,
        median_filter_size=5,
        dt=1 / 25,  # Temps entre deux mises à jour (~25 FPS)
    ):
        self.kf = cv2.KalmanFilter(9, 3)  # 9 états : position (x,y,z) + vitesse (vx,vy,vz) + accélération (ax,ay,az)

        # Matrice de mesure : on mesure seulement la position
        self.kf.measurementMatrix = np.zeros((3, 9), dtype=np.float32)
        self.kf.measurementMatrix[:3, :3] = np.eye(3)

        # Matrice de transition (modèle non linéaire avec vitesse et accélération)
        self.kf.transitionMatrix = np.eye(9, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = dt  # x dépend de v
            self.kf.transitionMatrix[i + 3, i + 6] = dt  # v dépend de a

        # Covariances des bruits
        self.kf.processNoiseCov = np.eye(9, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * error_cov_post

        self.initialized = False
        self.kf.statePre = np.zeros((9, 1), dtype=np.float32)
        self.kf.statePost = np.zeros((9, 1), dtype=np.float32)

        self.add_median_filter = add_median_filter
        if self.add_median_filter:
            self.median_filter = MedianFilter(median_filter_size)

    def update(self, measurement):
        """
        Met à jour la position avec une nouvelle mesure (x, y, z).
        Retourne la position lissée.
        """
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
    def __init__(self, camera: Orbbec, scale_percent: int = 100, up_mode=False):
        # orbbec initialization
        self.camera = camera
        self.scale_percent = scale_percent
        self.up_mode = up_mode

        # mediapipe holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
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

        # for visualization
        self.shoulders_visu = np.zeros((2, 3))
        self.user_center_visu = np.zeros(3)
        self.elbows_visu = np.zeros((2, 3))
        self.wrists_visu = np.zeros((2, 3))
        self.face_points_visu = np.zeros((len(FACELANDMARKS_CST), 3))

        # for the orientation of the hands
        self.left_hand_points_3D = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points_3D = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.kf_left_hand = [KalmanFilter3D(add_median_filter=False) for _ in range(len(HANDLANDMARKS_CST))]
        self.kf_right_hand = [KalmanFilter3D(add_median_filter=False) for _ in range(len(HANDLANDMARKS_CST))]

        self.orientation_hands = [np.eye(3) for _ in range(2)]

        self.dist_intershoulder = 300
        self.user_center = np.zeros(3)

        # parameters
        self.image_shape = self.camera.color_shape * self.scale_percent // 100
        self.camera_matrix = np.eye(3, dtype=np.float32)
        self.get_camera_matrix()

        self.T_world_camera = np.eye(3)

        # calibration if camera is upside
        if self.up_mode:
            self.calibrate()

    def get_camera_matrix(self):
        focal_length = self.image_shape[1]
        center = (self.image_shape[1] / 2, self.image_shape[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def calibrate(self, nb_frames=50):
        # récupérer la médiane des coordonnées 3D des épaules et du front de l'utilisateur pour déterminer l'orientation de la caméra
        shoulders_tab = np.zeros((nb_frames, 2, 3))
        forehead_tab = np.zeros((nb_frames, 3))
        ite = 0

        while ite < nb_frames:
            try:
                color_frame = self.camera.color_frame.pop()
                depth_frame = self.camera.depth_frame.pop()
            except Exception:
                continue

            image = self.resize_frames(color_frame, self.scale_percent)
            depth_frame = self.resize_frames(depth_frame, self.scale_percent)

            results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results:
                continue

            body_landmarks, face_landmarks = self.get_2D_landmarks(results)
            if body_landmarks is not None and face_landmarks is not None:
                [left_shoulder] = self.estimate_3D_landmark_with_depth(body_landmarks[SHOULDER_CST[0]], depth_frame)
                [right_shoulder] = self.estimate_3D_landmark_with_depth(body_landmarks[SHOULDER_CST[1]], depth_frame)
                dist_intershoulder = np.linalg.norm(left_shoulder - right_shoulder)
                [forehead] = self.estimate_3D_landmark_with_depth(face_landmarks[10], depth_frame)
                if left_shoulder is not None and right_shoulder is not None and forehead is not None:
                    for i in range(2):
                        left_shoulder[i] = left_shoulder[i] * 0.3 / dist_intershoulder
                        right_shoulder[i] = right_shoulder[i] * 0.3 / dist_intershoulder
                        forehead[i] = forehead[i] * 0.3 / dist_intershoulder
                    shoulders_tab[ite] = np.vstack((left_shoulder, right_shoulder))
                    forehead_tab[ite] = forehead
                    ite += 1

        shoulders_median = np.median(shoulders_tab, axis=0)
        forehead_median = np.median(forehead_tab, axis=0)

        angle = -np.arctan(
            (forehead_median[2] - np.mean(shoulders_median[:, 2]))
            / (forehead_median[1] - (np.mean(shoulders_median[:, 1])))
        )

        print(f"Camera orientation angle: {np.degrees(angle):.2f} ")

        self.T_world_camera = R.from_euler("x", angle, degrees=False).as_matrix()

    def resize_frames(self, frame, scale_percent=50):
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        return frame_resized

    def get_2D_landmarks(self, results):
        landmarks_dict = {"pose": None, "face": None}

        landmarks_sources = {
            "pose": results.pose_landmarks,
            "face": results.face_landmarks,
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
            z = original_landmarks[:, 2]

            x = np.clip(x_init, 0, self.image_shape[1] - 1)
            y = np.clip(y_init, 0, self.image_shape[0] - 1)
            landmarks_dict[hand] = np.vstack((x, y, z)).T.astype(np.float32)

        return [landmarks_dict["left"], landmarks_dict["right"]]

    def estimate_3D_landmark_with_depth(self, landmark, depth_frame, radius=2) -> Optional[np.ndarray]:
        if landmark is None:
            return None

        x, y = landmark[0], landmark[1]
        if (radius + 1 < x < self.image_shape[1] - radius - 1) and (radius + 1 < y < self.image_shape[0] - radius - 1):
            z = np.median(depth_frame[int(y - radius) : int(y + radius), int(x - radius) : int(x + radius)])
        else:
            z = depth_frame[int(y), int(x)]

        return np.vstack((x, y, z)).T.astype(np.float32)

    def get_landmarks_with_camera_transformation(self, landmarks, camera_upvalue=0):
        # get landmarks with camera transformation
        camera_upvalue_pixel = camera_upvalue * self.dist_intershoulder / 0.3
        translation_inverse = np.array([0, -camera_upvalue_pixel, 0])
        landmarks = np.reshape(landmarks, (3))
        landmarks[2] = landmarks[2] * self.dist_intershoulder / 0.3
        new_landmarks = np.dot(self.T_world_camera, (landmarks + translation_inverse))
        return new_landmarks

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

    def get_landmarks_coordinates(self, image, depth_frame, fixed_user=False) -> bool:
        image = self.resize_frames(image, self.scale_percent)
        depth_frame = self.resize_frames(depth_frame, self.scale_percent)
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results:
            return False

        body_landmarks, face_landmarks = self.get_2D_landmarks(results)
        left_hand_landmarks, right_hand_landmarks = self.get_3D_landmarks(results)

        if body_landmarks is not None:
            if not fixed_user:
                # get shoulder landmarks
                left_shoulder = self.estimate_3D_landmark_with_depth(body_landmarks[SHOULDER_CST[0]], depth_frame)
                if left_shoulder is not None:
                    self.shoulders_visu[0] = left_shoulder
                    self.shoulders[0] = self.get_landmarks_with_camera_transformation(left_shoulder)
                right_shoulder = self.estimate_3D_landmark_with_depth(body_landmarks[SHOULDER_CST[1]], depth_frame)
                if right_shoulder is not None:
                    self.shoulders_visu[1] = right_shoulder
                    self.shoulders[1] = self.get_landmarks_with_camera_transformation(right_shoulder)
                # get user center and distance intershoulder
                if np.any(self.shoulders[0]) and np.any(self.shoulders[1]):
                    self.user_center_visu = (self.shoulders_visu[0] + self.shoulders_visu[1]) / 2.0
                    self.user_center = (self.shoulders[0] + self.shoulders[1]) / 2.0
                    self.dist_intershoulder = np.linalg.norm(self.shoulders[0] - self.shoulders[1])

            # get elbows and wrists landmarks
            for i in range(2):
                elbow = self.estimate_3D_landmark_with_depth(body_landmarks[ELBOWS_CST[i]], depth_frame)
                if elbow is not None:
                    self.elbows_visu[i] = elbow
                    self.elbows[i] = self.get_landmarks_with_camera_transformation(elbow)
                wrist = self.estimate_3D_landmark_with_depth(body_landmarks[WRISTS_CST[i]], depth_frame, radius=10)
                if wrist is not None:
                    self.wrists_visu[i] = wrist
                    self.wrists[i] = self.get_landmarks_with_camera_transformation(wrist)

        for hand_landmarks, points in zip(
            [left_hand_landmarks, right_hand_landmarks], [self.left_hand_points, self.right_hand_points]
        ):
            if hand_landmarks is not None:
                for i, lm_cst in enumerate(HANDLANDMARKS_CST):
                    finger_coord = hand_landmarks[lm_cst]
                    if finger_coord is not None:
                        points[i] = self.get_landmarks_with_camera_transformation(finger_coord)

        if face_landmarks is not None:
            for i, lm_cst in enumerate(FACELANDMARKS_CST):
                face_coord = self.estimate_3D_landmark_with_depth(face_landmarks[lm_cst], depth_frame)
                if face_coord is not None:
                    self.face_points_visu[i] = face_coord
                    self.face_points[i] = self.get_landmarks_with_camera_transformation(face_coord)

        return True

    def visualization_landmarks(
        self,
        color_frame,
        left_goal_pose,
        right_goal_pose,
        roll,
        pitch,
        yaw,
        text_on=True,
        body_on=True,
        hands_on=True,
        face_on=True,
    ):

        color_frame = self.resize_frames(color_frame, self.scale_percent)
        cv2.circle(color_frame, (int(self.user_center_visu[0]), int(self.user_center_visu[1])), 5, (255, 255, 255), -1)
        if text_on:
            cv2.putText(
                color_frame,
                f"left_goal: {left_goal_pose[:3,3]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                color_frame,
                f"right_goal: {right_goal_pose[:3,3]}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if body_on:
            for i in range(2):
                cv2.circle(color_frame, (int(self.wrists_visu[i][0]), int(self.wrists_visu[i][1])), 5, (0, 255, 0), -1)
                cv2.circle(color_frame, (int(self.elbows_visu[i][0]), int(self.elbows_visu[i][1])), 5, (255, 0, 0), -1)
                cv2.circle(
                    color_frame, (int(self.shoulders_visu[i][0]), int(self.shoulders_visu[i][1])), 5, (0, 0, 255), -1
                )

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
                    (int(self.face_points_visu[i][0]), int(self.face_points_visu[i][1])),
                    5,
                    (255, 0, 255),
                    -1,
                )
            cv2.putText(
                color_frame,
                f"roll: {roll:.2f}, pitch: {pitch:.2f}, yaw: {yaw:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Color Viewer", color_frame)

    def init_plot(self):
        print("Init plot")
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, self.image_shape[1])
        ax.set_ylim(0, self.image_shape[0])
        ax.set_zlim(-500, 500)
        scatter = ax.scatter(0, 0, 0, c="r", marker="o")
        plt.show(block=False)
        return fig, ax, scatter

    def update_plot(self, ax, scatter):
        handpoints_3D = self.left_hand_points_3D

        colors = np.linspace(0, 1, len(handpoints_3D))  # Valeurs normalisées pour la colormap

        # Mettre à jour les points et leurs couleurs
        scatter.set_offsets(handpoints_3D[:, :2])  # Mise à jour X et Y
        scatter.set_3d_properties(handpoints_3D[:, 2], "z")  # Mise à jour Z
        scatter.set_array(colors)  # Mettre à jour les couleurs
        scatter.set_clim(0, 1)  # Fixe les bornes de la colormap
        plt.draw()
        plt.pause(0.001)


class RobotController:
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
        # self.rotation_smoother = [RotationSmoother(window_size=5), RotationSmoother(window_size=5)]

        self.head_rpy = deque(maxlen=10)

        self.rpy = np.zeros((2, 3))

    # def set_user_center(self, user_center):
    #     self.user_center = user_center

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

        # fig, ax, scatter = self.vision.init_plot()
        #  first pose
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
                        # self.robot_controller.reachy.l_arm.goto(left_goal_pose, duration=2)
                        # self.robot_controller.reachy.r_arm.goto(right_goal_pose, duration=2)
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

            self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)  # too long (0.025)
            # self.vision.update_plot(ax, scatter)

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

            # si échap ou ctrl c
            if self.raise_command_to_stop() or (cv2.waitKey(1) & 0xFF == 27):
                break
        # plt.ioff()
        # plt.show()
        cv2.destroyAllWindows()
        self.camera.stop()
        self.camera.frame_getter.join(5)


if __name__ == "__main__":
    camera = Orbbec()
    scale_percent = 50
    timestep = 0.02
    robot = RobotController("172.16.0.64")
    teleop = TeleopControl(camera, robot, scale_percent, timestep, mirror_mode=True)
    print("Teleoperation started")
    teleop.run()
