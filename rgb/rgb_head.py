import time
from collections import deque
from typing import List

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from pollen_vision.vision_models.object_detection import (  # type: ignore
    YoloWorldWrapper,
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
FACELANDMARKS_CST = [1, 152, 33, 263, 61, 291]
# Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner


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


class KalmanFilter2D:
    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=0.1,
        error_cov_post=1.0,
        add_median_filter=True,
        median_filter_size=5,
    ):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = (
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * process_noise
        )
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * measurement_noise
        self.kf.errorCovPost = (
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * error_cov_post
        )

        self.initialized = False
        self.kf.statePre = np.zeros((4, 1), dtype=np.float32)  # x, y, vx, vy
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

        self.add_median_filter = add_median_filter
        if self.add_median_filter:
            self.median_filter = MedianFilter(median_filter_size)

    def update(self, measurement):
        if measurement is None:
            return self.kf.statePost[:2].flatten() if self.initialized else None

        measurement = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePre[:2] = measurement
            self.kf.statePost[:2] = measurement
            self.initialized = True
            return measurement.flatten()

        self.kf.correct(measurement)
        prediction = self.kf.predict()
        filtered_position = np.array(prediction[:2]).flatten()
        if self.add_median_filter:
            filtered_position = self.median_filter.update(filtered_position)
        return filtered_position


class ComputerVision:
    def __init__(self):
        # mediapipe holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )

        # cv2
        cv2.setNumThreads(4)
        cv2.setUseOptimized(True)
        self.cap = cv2.VideoCapture(0)

        # ROI coordinates in image frame
        self.left_shoulder = None
        self.right_shoulder = None
        self.dist_intershoulder = None
        self.user_center = None
        self.face_points = np.zeros((len(FACELANDMARKS_CST), 2))

        self.kf_face = [KalmanFilter2D(median_filter_size=3) for _ in range(len(FACELANDMARKS_CST))]

        # parameters
        self.camera_matrix = np.zeros((3, 3))
        self.camera_calibration()

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None

    def get_xy_coordinates(self, landmarks, landmark_cst, image):
        if landmarks:
            landmarks = landmarks.landmark
            x = landmarks[landmark_cst].x * image.shape[1]
            y = landmarks[landmark_cst].y * image.shape[0]
            return np.array([x, y])
        else:
            return None

    def get_landmarks_coordinates(self, image, fixed_user=False) -> bool:
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results:
            if not fixed_user:
                self.left_shoulder = self.get_xy_coordinates(results.pose_landmarks, SHOULDER_CST[0], image)
                self.right_shoulder = self.get_xy_coordinates(results.pose_landmarks, SHOULDER_CST[1], image)
                if self.left_shoulder is not None and self.right_shoulder is not None:
                    self.user_center = (self.left_shoulder + self.right_shoulder) / 2.0
                    self.dist_intershoulder = np.linalg.norm(self.left_shoulder - self.right_shoulder)

            for i, lm_cst in enumerate(FACELANDMARKS_CST):
                face_coord = self.get_xy_coordinates(results.face_landmarks, lm_cst, image)
                if face_coord is not None:
                    self.face_points[i] = face_coord
            return True

        else:
            return False

    def camera_calibration(self):
        img = self.get_frame()
        focal_length = img.shape[1]
        center = (img.shape[1] / 2, img.shape[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def get_head_rotation(self):
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Nose tip
                [0.0, -330.0, -65.0],  # Chin
                [-175.0, 170.0, -135.0],  # Left eye left corner
                [175.0, 170.0, -135.0],  # Right eye right corner
                [-150.0, -150.0, -125.0],  # Left mouth corner
                [150.0, -150.0, -125.0],  # Right mouth corner
            ],
            dtype=np.float32,
        )

        dist_coeffs = np.zeros((4, 1))

        # get the filtered face points
        face_points_filtered = np.zeros((len(FACELANDMARKS_CST), 2))
        for idx in range(len(FACELANDMARKS_CST)):
            face_points_filtered[idx] = self.kf_face[idx].update(vision.face_points[idx])

        # Estimation de la pose via solvePnP
        success_pnp, rvec, tvec = cv2.solvePnP(
            model_points,
            face_points_filtered,
            vision.camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success_pnp:
            # Conversion du vecteur de rotation en matrice 3x3 puis en angles d'Euler
            rot, _ = cv2.Rodrigues(rvec)
            return rot

        return np.eye(3)

        #     # Conversion pour Reachy
        #     roll_reachy_frame = -yaw
        #     pitch_reachy_frame = roll - 180
        #     yaw_reachy_frame = -pitch

        # return roll_reachy_frame, pitch_reachy_frame, yaw_reachy_frame


class RobotController:
    def __init__(self, host):
        self.reachy = ReachySDK(host)
        self.reachy.turn_on()

    def convert_to_reachy_frame(self, rotation):
        # conversion de la matrice de rotation vers un nouveau repère où x = - ancien_z, y = ancien_x, z = - ancien_y
        # R_cam_to_reachy = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        # rotation_reachy_frame = np.dot(R_cam_to_reachy, rotation)
        # # en angles d'euler
        # roll_reachy_frame, pitch_reachy_frame, yaw_reachy_frame = R.from_matrix(rotation_reachy_frame).as_euler(
        #     "xyz", degrees=True
        # )
        roll, pitch, yaw = R.from_matrix(rotation).as_euler("xyz", degrees=True)
        roll_reachy_frame = -yaw
        pitch_reachy_frame = roll - 180
        yaw_reachy_frame = -pitch
        return [roll_reachy_frame, pitch_reachy_frame, yaw_reachy_frame]

    def set_head_orientation(self, rpy):
        roll, pitch, yaw = rpy
        self.reachy.head.neck.roll.goal_position = roll
        self.reachy.head.neck.pitch.goal_position = pitch
        self.reachy.head.neck.yaw.goal_position = yaw
        self.reachy.send_goal_positions(check_positions=False)


class TeleopControl:
    def __init__(self, vision: ComputerVision, robot_controller: RobotController):
        self.vision = vision
        self.robot_controller = robot_controller

    def run(self):
        while self.vision.cap.isOpened():
            img = self.vision.get_frame()
            if img is not None:
                self.vision.get_landmarks_coordinates(img)
                if np.any(self.vision.face_points):
                    head_rotation = self.vision.get_head_rotation()
                    rpy = self.robot_controller.convert_to_reachy_frame(head_rotation)
                    self.robot_controller.set_head_orientation(rpy)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    vision = ComputerVision()
    robot_controller = RobotController("localhost")
    teleop = TeleopControl(vision, robot_controller)
    teleop.run()
    vision.cap.release()
    cv2.destroyAllWindows()
