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

# original_landmarks constant :
HANDLANDMARKS_CST = [0, 2, 4, 5, 8, 9, 17, 20]
# wrist, thumb mcp, thumb tip, index mcp, index tip, middle mcp, pinky mcp, pinky tip

SHOULDER_CST = [11, 12]
ELBOWS_CST = [13, 14]
WRISTS_CST = [15, 16]
INDEX_MCP_CST = [19, 20]
THUMB_MCP_CST = [21, 22]
PINKY_MCP_CST = [17, 18]

FACELANDMARKS_CST = [1, 152, 33, 263, 61, 291]
# Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner


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
        process_noise=0.001,
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
    def __init__(self, camera: Orbbec, scale_percent: int = 100, up_mode=False):
        # orbbec initialization
        self.camera = camera
        self.scale_percent = scale_percent
        self.up_mode = up_mode

        # mediapipe holistic

        # BaseOptions = mp.tasks.BaseOptions
        # VisionRunningMode = mp.tasks.vision.RunningMode

        # options = vision.HolisticLandmarkerOptions(
        #     base_options=BaseOptions(model_asset_path="holistic_landmarker.task", delegate=1),
        #     running_mode=VisionRunningMode.VIDEO,
        # )

        # self.holistic = vision.HolisticLandmarker.create_from_options(options)
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
        self.wrists_3D = np.zeros((2, 3))
        self.pinky = np.zeros((2, 3))
        self.index = np.zeros((2, 3))
        self.thumb = np.zeros((2, 3))
        self.left_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.face_points = np.zeros((len(FACELANDMARKS_CST), 3))

        # for the orientation of the hands
        self.rpy_camera = [np.zeros(3), np.zeros(3)]
        self.left_hand_points_3D = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points_3D = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.kf_left_hand = [KalmanFilter3D(add_median_filter=False) for _ in range(len(HANDLANDMARKS_CST))]
        self.kf_right_hand = [KalmanFilter3D(add_median_filter=False) for _ in range(len(HANDLANDMARKS_CST))]

        self.orientation_hands = [np.eye(3) for _ in range(2)]

        self.dist_intershoulder = 0
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

            body_landmarks, face_landmarks, _, _ = self.get_2D_landmarks(results)
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

        # in Reachy frame, the orientation of the camera is around y-axis
        self.T_world_camera = R.from_euler("y", angle, degrees=False).as_matrix()

    def resize_frames(self, frame, scale_percent=50):
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        return frame_resized

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
        landmarks = results.pose_landmarks
        if landmarks is None:
            return None
        original_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
        y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)
        z = (original_landmarks[:, 2] * self.image_shape[1]).astype(int)

        x = np.clip(x_init, 0, self.image_shape[1] - 1)
        y = np.clip(y_init, 0, self.image_shape[0] - 1)
        return np.vstack((x, y, z)).T.astype(np.float32)
        # hands = [
        #     ("left", results.left_hand_landmarks),
        #     ("right", results.right_hand_landmarks),
        # ]

        # landmarks_dict = {"left": None, "right": None, "face": None}

        # for hand, original_landmarks in hands:
        #     if original_landmarks is None:
        #         continue

        #     original_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in original_landmarks.landmark])
        #     x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
        #     y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)
        #     z = original_landmarks[:, 2]

        #     x = np.clip(x_init, 0, self.image_shape[1] - 1)
        #     y = np.clip(y_init, 0, self.image_shape[0] - 1)
        #     landmarks_dict[hand] = np.vstack((x, y, z)).T.astype(np.float32)

        # return [landmarks_dict["left"], landmarks_dict["right"]]

    def estimate_3D_landmark_with_depth(self, landmark, depth_frame, radius=2) -> Optional[np.ndarray]:
        if landmark is None:
            return None

        x, y = landmark[0], landmark[1]
        if (radius + 1 < x < self.image_shape[1] - radius - 1) and (radius + 1 < y < self.image_shape[0] - radius - 1):
            z = np.median(depth_frame[int(y - radius) : int(y + radius), int(x - radius) : int(x + radius)])
        else:
            z = depth_frame[int(y), int(x)]

        return np.vstack((x, y, z)).T.astype(np.float32)

        # z = np.zeros(len(x))
        # radius = 1

        # for i in range(len(x)):
        #     if (
        #         radius + 1 < x[i] < self.image_shape[1] - radius - 1
        #         and radius + 1 < y[i] < self.image_shape[0] - radius - 1
        #     ):
        #         z[i] = np.median(depth_frame[y[i] - radius : y[i] + radius, x[i] - radius : x[i] + radius])
        #     else:
        #         z[i] = depth_frame[y[i], x[i]]

        # return np.vstack((x, y, z)).T.astype(np.float32)

    def get_rpy_from_vector(self, side_int, index, wrist):
        vector = index - wrist
        vector_xy = np.array([vector[0], vector[1], 0])
        norm_xy = np.linalg.norm(vector_xy)

        vector_xz = np.array([vector[0], 0, vector[2]])
        norm_xz = np.linalg.norm(vector_xz)

        angle_xy = np.arccos(vector[0] / norm_xy) * 180 / np.pi if norm_xy != 0 else 0
        angle_xz = np.arccos(vector[0] / norm_xz) * 180 / np.pi if norm_xz != 0 else 0

        # angle_plans = self.angle_between_two_planes()

        print("Yaw: {:.2f}°".format(angle_xy))
        print("Pitch: {:.2f}°".format(angle_xz))

        self.rpy_camera[side_int] = np.array([angle_xy, 0, angle_xz])

    def angle_between_vector_and_plane(self, vector, plane_normal):
        vector = vector / np.linalg.norm(vector)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        cos_theta = np.dot(vector, plane_normal)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Évite les erreurs numériques

        angle_with_plane = np.degrees(90 - theta_rad)

        return angle_with_plane

    def angle_between_two_planes(self, vector1, vector2, plane_normal):
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        plane2_normal = np.cross(vector1, vector2)
        plane2_normal = plane2_normal / np.linalg.norm(plane2_normal)

        cos_theta = np.dot(plane2_normal, plane_normal)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_with_plane = np.degrees(theta_rad)
        return angle_with_plane

    def get_hand_orientation(self, pose_landmarks):
        if pose_landmarks is None:
            return
        wrists = [pose_landmarks[WRISTS_CST[0]], pose_landmarks[WRISTS_CST[1]]]
        index = [pose_landmarks[INDEX_MCP_CST[0]], pose_landmarks[INDEX_MCP_CST[1]]]
        thumb = [pose_landmarks[THUMB_MCP_CST[0]], pose_landmarks[THUMB_MCP_CST[1]]]
        pinky = [pose_landmarks[PINKY_MCP_CST[0]], pose_landmarks[PINKY_MCP_CST[1]]]

        for side_int in range(2):
            self.get_rpy_from_vector(side_int, index[side_int], wrists[side_int])
            self.index[side_int] = index[side_int]
            self.thumb[side_int] = thumb[side_int]
            self.wrists_3D[side_int] = wrists[side_int]
            self.pinky[side_int] = pinky[side_int]

    # def get_hand_orientation(self, handlandmarks_both_side):
    # for side_int in range(2):
    #     kf_hand = self.kf_left_hand if side_int == 0 else self.kf_right_hand
    #     handpoints = self.left_hand_points_3D if side_int == 0 else self.right_hand_points_3D
    #     handlandmarks = handlandmarks_both_side[side_int]
    #     if handlandmarks is not None:
    #         for ite, lm_cst in enumerate(HANDLANDMARKS_CST):
    #             finger_coord = handlandmarks[lm_cst]
    #             if finger_coord is not None:
    #                 coord_filtered = kf_hand[ite].update(finger_coord)
    #                 handpoints[ite] = coord_filtered
    #         if side_int == 0:
    #             self.left_hand_points_3D = handpoints
    #         else:
    #             self.right_hand_points_3D = handpoints

    #     wrist = handpoints[0]
    #     thumb_mcp = handpoints[1]
    #     index_mcp = handpoints[3]
    #     thumb_tip = handpoints[2]
    #     index_tip = handpoints[4]

    #     self.get_rpy_from_vector(side_int, index_mcp, wrist)

    # # définition des vecteurs
    # vector_index = index_mcp - wrist
    # vector_thumb = thumb_mcp - wrist

    # # calcul des angles
    # yaw_camera = self.angle_between_vector_and_plane(
    #     vector_index, np.array([1, 0, 0])
    # )  # index / plan vertical
    # roll_camera = self.angle_between_vector_and_plane(
    #     vector_index, np.array([0, 1, 0])
    # )  # index / plan horizontal
    # pitch_camera = self.angle_between_two_planes(
    #     vector_index, vector_thumb, np.array([0, 1, 0])
    # )  # plan index/pouce / plan horizontal

    # print(f"yaw: {yaw_camera:.2f}, roll: {roll_camera:.2f}, pitch: {pitch_camera:.2f}")
    # self.rpy_camera[side_int] = np.array([roll_camera, pitch_camera, yaw_camera])

    def get_landmarks_coordinates(self, image, depth_frame, fixed_user=False, closed_fists=False) -> bool:
        image = self.resize_frames(image, self.scale_percent)
        depth_frame = self.resize_frames(depth_frame, self.scale_percent)
        # results = self.holistic.detect_for_video(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results:
            print("No results")
            return False

        if closed_fists:
            effector_cst = WRISTS_CST  # INDEX_MCP_CST
        else:
            effector_cst = WRISTS_CST

        body_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks = self.get_2D_landmarks(results)

        pose_landmarks = self.get_3D_landmarks(results)
        print("get3Dlandmarks")
        self.get_hand_orientation(pose_landmarks)
        print("gethandorientation")

        # handpoints_3D = self.get_3D_landmarks(results)
        # self.get_hand_orientation(handpoints_3D)

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
                wrist = self.estimate_3D_landmark_with_depth(body_landmarks[effector_cst[i]], depth_frame, radius=10)
                if wrist is not None:
                    self.wrists[i] = wrist

        ite = 0
        for hand_landmarks, points in zip(
            [left_hand_landmarks, right_hand_landmarks], [self.left_hand_points, self.right_hand_points]
        ):
            if hand_landmarks is not None:
                for i, lm_cst in enumerate(HANDLANDMARKS_CST):
                    finger_coord = self.estimate_3D_landmark_with_depth(hand_landmarks[lm_cst], depth_frame)
                    if finger_coord is not None:
                        points[i] = finger_coord
                # if hand_landmarks[0][:2] is not None:
                #     self.wrists[ite] = self.estimate_3D_landmark_with_depth(hand_landmarks[0][:2], depth_frame)
            ite += 1

        if face_landmarks is not None:
            for i, lm_cst in enumerate(FACELANDMARKS_CST):
                face_coord = self.estimate_3D_landmark_with_depth(face_landmarks[lm_cst], depth_frame)
                if face_coord is not None:
                    self.face_points[i] = face_coord

        return True

    def visualization_landmarks(
        self,
        color_frame,
        left_goal_pose,
        right_goal_pose,
        roll,
        pitch,
        yaw,
        body_on=True,
        hands_on=True,
        face_on=True,
        text_on=True,
    ):
        color_frame = self.resize_frames(color_frame, self.scale_percent)
        cv2.circle(color_frame, (int(self.user_center[0]), int(self.user_center[1])), 5, (255, 255, 255), -1)
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
                cv2.circle(color_frame, (int(self.wrists[i][0]), int(self.wrists[i][1])), 5, (0, 255, 0), -1)
                cv2.circle(color_frame, (int(self.elbows[i][0]), int(self.elbows[i][1])), 5, (255, 0, 0), -1)
                cv2.circle(color_frame, (int(self.shoulders[i][0]), int(self.shoulders[i][1])), 5, (0, 0, 255), -1)
                cv2.circle(color_frame, (int(self.index[i][0]), int(self.index[i][1])), 5, (255, 255, 0), -1)
                cv2.circle(color_frame, (int(self.thumb[i][0]), int(self.thumb[i][1])), 5, (0, 255, 255), -1)

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
        handpoints_3D = []
        handpoints_3D.append(self.wrists_3D[0])
        handpoints_3D.append(self.elbows[0])
        handpoints_3D.append(self.shoulders[0])
        handpoints_3D.append(self.index[0])
        handpoints_3D.append(self.thumb[0])
        handpoints_3D = np.array(handpoints_3D)

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

        self.mf_gripper = [MedianFilter(), MedianFilter()]
        self.rotation_smoother = [RotationSmoother(window_size=5), RotationSmoother(window_size=5)]

        self.former_poses = [
            self.reachy.l_arm.forward_kinematics(),
            self.reachy.r_arm.forward_kinematics(),
        ]

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

    def convert_rotation_to_reachy_frame(self, rotation_matrix):
        # conversion dans le repère de Reachy (x = -z, y = x, z = -y)
        T_cam_to_reachy = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

        # Appliquer la correction après transformation
        rotation_matrix_reachy = T_cam_to_reachy @ rotation_matrix

        return rotation_matrix_reachy

    def correct_position_with_camera_orientation(self, rotation_matrix, goal_position):
        return np.dot(rotation_matrix, goal_position)

    def is_top_grasp_pose(self, elbow, user_center, dist_intershoulder):
        if elbow is None:
            return False
        return elbow[1] < user_center[1] + dist_intershoulder / 2

    def get_effector_pose(self, side, vision, mode="shoulder", filter=True):
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
        # goal_position_corrected = self.correct_position_with_camera_orientation(vision.T_world_camera, goal_position)
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
            elbow_new_depth = self.estimate_elbow_depth(
                self.real_shoulders[side_int], goal_position_filtered, elbow_position_filtered
            )
            vect = goal_position_filtered - elbow_new_depth
            vect = vect / np.linalg.norm(vect)
            rotation_matrix = rotation_matrix_from_vector(vect)

        if mode == "hand":

            rotation_matrix_hand = self.convert_rotation_to_reachy_frame(vision.orientation_hands[side_int])
            rotation_matrix = self.rotation_smoother[side_int].update(rotation_matrix_hand)

        goal_pose = recompose_matrix(rotation_matrix, np.round(goal_position_filtered, 3))

        # if filter and self.is_too_far(goal_pose, side_int):
        #     print("Goal pose is too far")
        #     return self.former_poses[side_int]

        self.former_poses[side_int] = goal_pose

        return goal_pose

    def is_too_far(self, goal_pose, side_int):
        goal_position = goal_pose[:3, 3]
        former_position = self.former_poses[side_int][:3, 3]
        return np.linalg.norm(goal_position - former_position) > 0.2

    # def normalize_vector(self, v):
    #     """Normalise un vecteur et gère le cas où la norme est zéro."""
    #     norm = np.linalg.norm(v)
    #     if norm == 0:
    #         return np.array([0.0, 0.0, 0.0])  # Retourne un vecteur nul pour éviter NaN
    #     return v / norm

    def estimate_elbow_depth(self, shoulder_3D, wrist_3D, elbow):
        x_e, y_e, z_e = shoulder_3D[0], shoulder_3D[1], shoulder_3D[2]
        y_c, z_c = elbow[1], elbow[2]  # Coordonnées 2D détectées du coude

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

    # def make_line(self, end_pose: list[npt.NDArray[np.float64]], duration: float):
    #     t0 = time.time()
    #     start_pose = self.former_poses
    #     t1 = time.time()

    #     start_pose = [
    #         self.reachy.l_arm.forward_kinematics(),
    #         self.reachy.r_arm.forward_kinematics(),
    #     ]
    #     t1bis = time.time()
    #     print(f"t before getting poses: {t1 - t0} and {t1bis - t1}")
    #     start_position = [start_pose[0][:3, 3], start_pose[1][:3, 3]]
    #     end_position = [end_pose[0][:3, 3], end_pose[1][:3, 3]]
    #     start_rotation = [
    #         R.from_matrix(start_pose[0][:3, :3]),
    #         R.from_matrix(start_pose[1][:3, :3]),
    #     ]
    #     end_rotation = [
    #         R.from_matrix(end_pose[0][:3, :3]),
    #         R.from_matrix(end_pose[1][:3, :3]),
    #     ]
    #     t2 = time.time()
    #     print(f"t after getting poses: {t2 - t1}")

    #     nbr_points = int(duration * self.control_frequency)

    #     left_slerp = Slerp(
    #         [0, 1],
    #         R.from_matrix([start_rotation[0].as_matrix(), end_rotation[0].as_matrix()]),
    #     )
    #     right_slerp = Slerp(
    #         [0, 1],
    #         R.from_matrix([start_rotation[1].as_matrix(), end_rotation[1].as_matrix()]),
    #     )
    #     t3 = time.time()
    #     print(f"t before sending commands: {t3 - t2}")

    #     for i in range(nbr_points):
    #         t = time.time()
    #         alpha = i / nbr_points

    #         left_interp_rotation = left_slerp([alpha]).as_matrix()[0]
    #         left_interp_position = start_position[0] + alpha * (end_position[0] - start_position[0])

    #         right_interp_rotation = right_slerp([alpha]).as_matrix()[0]
    #         right_interp_position = start_position[1] + alpha * (end_position[1] - start_position[1])

    #         left_pose = recompose_matrix(left_interp_rotation, left_interp_position)
    #         right_pose = recompose_matrix(right_interp_rotation, right_interp_position)

    #         self.go_to_pose(left_pose, "l_arm")
    #         self.go_to_pose(right_pose, "r_arm")
    #         time.sleep(max(1.0 / self.control_frequency - (time.time() - t), 0.0))

    #     self.former_poses = end_pose
    #     t4 = time.time()
    #     print(f"t after sending commands: {t4 - t3}")

    def get_head_rotation(self, vision):
        face_points_filtered = np.zeros((len(FACELANDMARKS_CST), 3))
        for idx in range(len(FACELANDMARKS_CST)):
            face_point_filtered = self.kf_face[idx].update(vision.face_points[idx])
            face_point_reachy_frame = self.convert_to_robot_frame(
                face_point_filtered, vision.user_center, vision.dist_intershoulder
            )
            face_points_filtered[idx] = face_point_reachy_frame
            if vision.up_mode:
                face_points_filtered[idx] = self.correct_position_with_camera_orientation(
                    vision.T_world_camera, face_point_reachy_frame
                )

        left_eye = face_points_filtered[2]
        right_eye = face_points_filtered[3]
        nose = face_points_filtered[0]
        chin = face_points_filtered[1]

        y_axis = left_eye - right_eye
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = nose - chin
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        rot = np.array([x_axis, y_axis, z_axis]).T
        roll, pitch, yaw = R.from_matrix(rot).as_euler("XYZ", degrees=True)

        return roll, pitch - 40, 180 - yaw

    def set_head_orientation(self, roll, pitch, yaw):
        self.reachy.head.neck.roll.goal_position = roll
        self.reachy.head.neck.pitch.goal_position = pitch
        self.reachy.head.neck.yaw.goal_position = yaw
        self.reachy.send_goal_positions(check_positions=False)

    def get_gripper_command(self, side, vision, closed_fists=False):
        if side == "left":
            hand_points = vision.left_hand_points
            kf_hand = self.kf_left_hand
            gripper = self.reachy.l_arm.gripper
            side_int = 0
        else:
            hand_points = vision.right_hand_points
            kf_hand = self.kf_right_hand
            gripper = self.reachy.r_arm.gripper
            side_int = 1

        # get the filtered fingers position
        hand_points_filtered = np.zeros((len(HANDLANDMARKS_CST), 3))
        for idx in range(len(HANDLANDMARKS_CST)):
            hand_points_filtered[idx] = kf_hand[idx].update(hand_points[idx])

        index = hand_points_filtered[4]
        thumb = hand_points_filtered[2]
        pinky = hand_points_filtered[6]

        if not closed_fists:
            dist_index_thumb_normalized = np.linalg.norm(index - thumb) / vision.dist_intershoulder
            dist_filtered = self.mf_gripper[side_int].update(dist_index_thumb_normalized)
            if not gripper.is_moving() and dist_filtered < 0.1:
                gripper.close()

            elif not gripper.is_moving() and dist_filtered > 0.3:
                gripper.open()

        else:
            dist_finger_normalized = (
                np.max([np.linalg.norm(index - pinky), np.linalg.norm(thumb - pinky)]) / vision.dist_intershoulder
            )
            dist_filtered = self.mf_gripper[side_int].update(dist_finger_normalized)
            print("side ", dist_filtered)
            if not gripper.is_moving() and self.grippers_ready[side_int] and dist_filtered > 0.4:
                if gripper.get_current_opening() < 50:
                    gripper.open()
                else:
                    gripper.close()
                self.grippers_ready[side_int] = False

            elif not gripper.is_moving() and dist_filtered < 0.2:
                self.grippers_ready[side_int] = True


class TeleopControl:
    def __init__(self, camera: Orbbec, robot_controller: RobotController, scale_percent: int, timestep: float):
        self.camera = camera
        self.vision = ComputerVision(self.camera, scale_percent)
        self.robot_controller = robot_controller
        self.timestep = timestep
        self.first_pose_done = False

    def run(self):
        fig, ax, scatter = self.vision.init_plot()
        #  first pose
        while not self.first_pose_done:
            try:
                color_frame = self.camera.color_frame.pop()
                depth_frame = self.camera.depth_frame.pop()

                self.vision.get_landmarks_coordinates(color_frame, depth_frame, fixed_user=False)
                if np.any(self.vision.user_center) and np.any(self.vision.wrists):
                    left_goal_pose = self.robot_controller.get_effector_pose("left", self.vision, filter=False)
                    right_goal_pose = self.robot_controller.get_effector_pose("right", self.vision, filter=False)
                    self.robot_controller.go_to_pose(left_goal_pose, "l_arm")
                    self.robot_controller.go_to_pose(right_goal_pose, "r_arm")
                    # self.robot_controller.make_line([left_goal_pose, right_goal_pose], self.timestep)  # too long (0.036)
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
            self.vision.update_plot(ax, scatter)

            # make the robot follow the user's hands
            left_goal_pose = self.robot_controller.get_effector_pose("left", self.vision, "elbow")
            right_goal_pose = self.robot_controller.get_effector_pose("right", self.vision, "elbow")
            self.robot_controller.go_to_pose(left_goal_pose, "l_arm")
            self.robot_controller.go_to_pose(right_goal_pose, "r_arm")

            roll, pitch, yaw = 0, 0, 0
            # make the robot follow the user's head
            if np.any(self.vision.face_points):
                roll, pitch, yaw = self.robot_controller.get_head_rotation(self.vision)
                self.robot_controller.set_head_orientation(roll, pitch, yaw)

            # make the robot grab the objects
            self.robot_controller.get_gripper_command("left", self.vision)
            self.robot_controller.get_gripper_command("right", self.vision)

            cv2.putText(
                color_frame,
                f"LFT : {np.round(self.vision.rpy_camera[0],3)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                color_frame,
                f"RGT : {np.round(self.vision.rpy_camera[1],3)}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # to visualize original_landmarks
            self.vision.visualization_landmarks(
                color_frame,
                left_goal_pose,
                right_goal_pose,
                roll,
                pitch,
                yaw,
                body_on=True,
                hands_on=False,
                face_on=False,
                text_on=False,
            )
            print(f"Freq: {1/(time.time() - t):.3f}")

            # si échap ou ctrl c
            if cv2.waitKey(1) & 0xFF == 27:
                break
        plt.ioff()
        plt.show()
        cv2.destroyAllWindows()
        self.camera.stop()
        self.camera.frame_getter.join(5)


if __name__ == "__main__":
    camera = Orbbec()
    scale_percent = 50
    timestep = 0.02
    robot = RobotController("localhost")
    teleop = TeleopControl(camera, robot, scale_percent, timestep)
    teleop.run()
