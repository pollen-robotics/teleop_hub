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

# landmarks constant :
INDEX_CST = 8
LEFT_SHOULDER_CST = 11
RIGHT_SHOULDER_CST = 12
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
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = (
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            * process_noise
        )
        self.kf.measurementNoiseCov = (
            np.array([[1, 0], [0, 1]], np.float32) * measurement_noise
        )
        self.kf.errorCovPost = (
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            * error_cov_post
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
    def __init__(self, object_labels, detection_threshold):
        # mediapipe holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        self.results = None

        # pollen vision
        self.object_detection = YoloWorldWrapper()
        self.object_labels = object_labels
        self.detection_threshold = detection_threshold

        # cv2
        cv2.setNumThreads(4)
        cv2.setUseOptimized(True)
        self.cap = cv2.VideoCapture(0)

        # ROI coordinates in image frame
        self.left_object_dict = {}
        self.right_object_dict = {}
        self.left_shoulder = None
        self.right_shoulder = None
        self.dist_intershoulder = None
        self.user_center = None
        self.left_index = None
        self.right_index = None
        self.face_points = None

        # parameters
        self.camera_matrix = np.zeros((3, 3))

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # self.results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # if self.results.pose_landmarks:
            return frame
        else:
            # self.results = None
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
                self.left_shoulder = self.get_xy_coordinates(
                    results.pose_landmarks, LEFT_SHOULDER_CST, image
                )
                self.right_shoulder = self.get_xy_coordinates(
                    results.pose_landmarks, RIGHT_SHOULDER_CST, image
                )
                if self.left_shoulder is not None and self.right_shoulder is not None:
                    self.user_center = (self.left_shoulder + self.right_shoulder) / 2.0
                    self.dist_intershoulder = np.linalg.norm(
                        self.left_shoulder - self.right_shoulder
                    )

            self.left_index = self.get_xy_coordinates(
                results.left_hand_landmarks, INDEX_CST, image
            )
            self.right_index = self.get_xy_coordinates(
                results.right_hand_landmarks, INDEX_CST, image
            )

            face_points = []
            for idx in FACELANDMARKS_CST:
                face_coord = self.get_xy_coordinates(results.face_landmarks, idx, image)
                if face_coord is not None:
                    face_points.append(face_coord.tolist())
            self.face_points = np.array(face_points, dtype=np.float32)
            return True

        else:
            return False

    def get_object_dict(self, image):
        predictions = self.object_detection.infer(
            im=image,
            candidate_labels=self.object_labels,
            detection_threshold=self.detection_threshold,
        )
        highest_score_objects = {}
        for obj in predictions:
            label = obj.get("label")
            if label:
                if (
                    label not in highest_score_objects
                    or obj["score"] > highest_score_objects[label]["score"]
                ):
                    highest_score_objects[label] = obj

        left_object_prediction = highest_score_objects.get(self.object_labels[0], None)
        right_object_prediction = highest_score_objects.get(self.object_labels[1], None)

        # capture the center and the diameter of the objects
        if left_object_prediction:
            self.left_object_dict["center"] = np.array(
                [
                    (
                        left_object_prediction["box"]["xmin"]
                        + left_object_prediction["box"]["xmax"]
                    )
                    / 2,
                    (
                        left_object_prediction["box"]["ymin"]
                        + left_object_prediction["box"]["ymax"]
                    )
                    / 2,
                ]
            )
            self.left_object_dict["diameter"] = np.mean(
                [
                    left_object_prediction["box"]["xmax"]
                    - left_object_prediction["box"]["xmin"],
                    left_object_prediction["box"]["ymax"]
                    - left_object_prediction["box"]["ymin"],
                ]
            )
        else:
            self.left_object_dict = {}

        if right_object_prediction:
            self.right_object_dict["center"] = np.array(
                [
                    (
                        right_object_prediction["box"]["xmin"]
                        + right_object_prediction["box"]["xmax"]
                    )
                    / 2,
                    (
                        right_object_prediction["box"]["ymin"]
                        + right_object_prediction["box"]["ymax"]
                    )
                    / 2,
                ]
            )
            self.right_object_dict["diameter"] = np.mean(
                [
                    right_object_prediction["box"]["xmax"]
                    - right_object_prediction["box"]["xmin"],
                    right_object_prediction["box"]["ymax"]
                    - right_object_prediction["box"]["ymin"],
                ]
            )
        else:
            self.right_object_dict = {}

    def get_depth_parameters(self, left_diam, right_diam, dists_inter_shoulder):
        reachy_arm_straight = 0.7
        reachy_arm_bent = 0.2
        diam_forward, diam_backward = np.zeros(2), np.zeros(2)
        dist_inter_shoulder = np.mean(dists_inter_shoulder)
        diam_forward[0] = np.max(left_diam) / dist_inter_shoulder
        diam_forward[1] = np.max(right_diam) / dist_inter_shoulder
        diam_backward[0] = np.min(left_diam) / dist_inter_shoulder
        diam_backward[1] = np.min(right_diam) / dist_inter_shoulder
        a, b = np.zeros(2), np.zeros(2)

        for i in range(2):
            a[i] = (reachy_arm_straight - reachy_arm_bent) / (
                diam_forward[i] - diam_backward[i]
            )
            b[i] = reachy_arm_straight - a[i] * diam_forward[i]

        self.a = a
        self.b = b

    def camera_calibration(self, img):
        focal_length = img.shape[1]
        center = (img.shape[1] / 2, img.shape[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def calibrate(self) -> None:
        print("Calibrating...")
        starting_time = time.time()
        left_diam: List[float] = []
        right_diam: List[float] = []
        dists_inter_shoulder: List[float] = []

        while (
            (time.time() - starting_time < 5)
            or (len(left_diam) < 10)
            or (len(right_diam) < 10)
            or (len(dists_inter_shoulder) < 10)
        ):
            img = self.get_frame()

            if img is not None:
                # balls detection
                self.get_object_dict(img)
                if self.left_object_dict:
                    left_diam.append(self.left_object_dict["diameter"])
                if self.right_object_dict:
                    right_diam.append(self.right_object_dict["diameter"])

                # landmarks detection
                success = self.get_landmarks_coordinates(img)
                if success and self.dist_intershoulder:
                    dists_inter_shoulder.append(self.dist_intershoulder)

        self.get_depth_parameters(left_diam, right_diam, dists_inter_shoulder)
        self.camera_calibration(img)
        print("Calibration done.")


class RobotController:
    def __init__(self, host):
        self.reachy = ReachySDK(host)
        self.reachy.turn_on()

        # user parameters
        self.real_shoulders = np.array([[0, 0.15, 0], [0, -0.15, 0]])
        self.real_dist_intershoulder = 0.3

        # control parameters
        self.control_frequency = 100
        self.grippers_ready = [
            self.reachy.l_arm.gripper.is_on(),
            self.reachy.r_arm.gripper.is_on(),
        ]

        # filters
        self.kf_left_object = KalmanFilter2D()
        self.kf_right_object = KalmanFilter2D()
        self.kf_left_index = KalmanFilter2D(add_median_filter=False)
        self.kf_right_index = KalmanFilter2D(add_median_filter=False)
        self.kf_face = [
            KalmanFilter2D(median_filter_size=3) for _ in range(len(FACELANDMARKS_CST))
        ]

    def estimate_depth(self, side, vision):
        side_int = 0 if side == "left" else 1
        object_diameter = (
            vision.left_object_dict["diameter"]
            if side == "left"
            else vision.right_object_dict["diameter"]
        )
        return (
            vision.a[side_int] * object_diameter / vision.dist_intershoulder
            + vision.b[side_int]
        )

    def convert_to_robot_frame(
        self, obj_center, user_center, dist_intershoulder, depth
    ):
        position2D_user_frame = (
            (obj_center - user_center)
            * self.real_dist_intershoulder
            / dist_intershoulder
        )
        position3D_robot_frame = np.array(
            [depth, position2D_user_frame[0], -position2D_user_frame[1]]
        )
        return position3D_robot_frame

    def get_effector_pose(self, side, vision):
        user_center = vision.user_center
        dist_intershoulder = vision.dist_intershoulder
        side_int = 0 if side == "left" else 1

        # get the filtered object center
        if side_int == 0:
            obj_center = self.kf_left_object.update(vision.left_object_dict["center"])
        else:
            obj_center = self.kf_right_object.update(vision.right_object_dict["center"])

        depth = self.estimate_depth(side, vision)

        goal_position = self.convert_to_robot_frame(
            obj_center, user_center, dist_intershoulder, depth
        )
        vect = goal_position - self.real_shoulders[side_int]
        vect = vect / np.linalg.norm(vect)
        rotation_matrix = rotation_matrix_from_vector(vect)
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
            left_interp_position = start_position[0] + alpha * (
                end_position[0] - start_position[0]
            )

            right_interp_rotation = right_slerp([alpha]).as_matrix()[0]
            right_interp_position = start_position[1] + alpha * (
                end_position[1] - start_position[1]
            )

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

        # get the filtered face points
        face_points_filtered = np.zeros((len(FACELANDMARKS_CST), 2))
        for idx in range(len(FACELANDMARKS_CST)):
            face_points_filtered[idx] = self.kf_face[idx].update(
                vision.face_points[idx]
            )

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
            roll, pitch, yaw = rotationMatrixToEulerAngles(rot)

            # Conversion pour Reachy
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
        obj = vision.left_object_dict if side == "left" else vision.right_object_dict

        # get the filtered index position
        if side == "left":
            index = self.kf_left_index.update(vision.left_index)
        else:
            index = self.kf_right_index.update(vision.right_index)

        gripper = (
            self.reachy.l_arm.gripper if side == "left" else self.reachy.r_arm.gripper
        )
        side_int = 0 if side == "left" else 1

        if not obj or index is None:
            return

        try:
            dist_index_ball = np.linalg.norm(index - obj["center"])
            dist_normalized = dist_index_ball / (obj["diameter"] / 2)

            if dist_normalized > 1.5 and self.grippers_ready[side_int]:
                self.grippers_ready[side_int] = False
                (
                    gripper.close()
                    if gripper.get_current_opening() > 0.5
                    else gripper.open()
                )

            elif dist_normalized < 1.1 and not self.grippers_ready[side_int]:
                self.grippers_ready[side_int] = True
        except Exception:
            print(f"Error in get_gripper_command : {obj} and {index} ")


class TeleopControl:
    def __init__(
        self, vision: ComputerVision, robot_controller: RobotController, timestep: float
    ):
        self.vision = vision
        self.robot_controller = robot_controller
        self.timestep = timestep
        self.first_pose_done = False

    def run(self):
        # initialization and calibration
        self.vision.calibrate()

        # first pose
        img = self.vision.get_frame()
        self.vision.get_landmarks_coordinates(img)
        while not self.first_pose_done:
            img = self.vision.get_frame()
            if img is not None:
                self.vision.get_object_dict(img)
                success = self.vision.get_landmarks_coordinates(img)

                if (
                    success
                    and self.vision.left_object_dict
                    and self.vision.right_object_dict
                ):
                    left_goal_pose = self.robot_controller.get_effector_pose(
                        "left", self.vision
                    )
                    right_goal_pose = self.robot_controller.get_effector_pose(
                        "right", self.vision
                    )
                    self.robot_controller.make_line(
                        [left_goal_pose, right_goal_pose], 2.0
                    )
                    self.first_pose_done = True
                    print("First pose done")

        # teleoperation
        while self.vision.cap.isOpened():
            img = self.vision.get_frame()
            if img is not None:
                self.vision.get_landmarks_coordinates(img, fixed_user=True)
                self.vision.get_object_dict(img)

                # make the robot follow the user's hands
                if self.vision.left_object_dict:
                    left_goal_pose = self.robot_controller.get_effector_pose(
                        "left", self.vision
                    )
                if self.vision.right_object_dict:
                    right_goal_pose = self.robot_controller.get_effector_pose(
                        "right", self.vision
                    )
                self.robot_controller.make_line(
                    [left_goal_pose, right_goal_pose], self.timestep
                )

                # make the robot follow the user's head
                roll, pitch, yaw = self.robot_controller.get_head_rotation(vision)
                self.robot_controller.set_head_orientation(roll, pitch, yaw)

                # make the robot grab the objects
                self.robot_controller.get_gripper_command("left", vision)
                self.robot_controller.get_gripper_command("right", vision)

            if cv2.waitKey(5) & 0xFF == 27:
                break


if __name__ == "__main__":
    vision = ComputerVision(["tennis ball or yellow ball", "orange ball"], 0.2)
    robot = RobotController("localhost")
    teleop = TeleopControl(vision, robot, 0.05)
    teleop.run()
