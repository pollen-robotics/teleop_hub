from typing import Optional

import cv2  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
from controller.filter.kalman_filter import KalmanFilter3D  # type: ignore
from controller.rgbd_tracker.orbbec import Orbbec  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from utils import normalize_vector

SHOULDER_CST = [11, 12]
ELBOWS_CST = [13, 14]
WRISTS_CST = [19, 20]

FACELANDMARKS_CST = [1, 152, 33, 263]  # Nose tip, Chin, Left eye left corner, Right eye right corner
HANDLANDMARKS_CST = [4, 8]  # thumb tip, index tip


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

        self.dist_intershoulder = 300.0
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
        # get camera orientation from the difference between the shoulders and the forehead depth
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
            x_axis = normalize_vector(x_axis)
            z_axis = wrist - mid_index_thumb
            z_axis = normalize_vector(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = normalize_vector(y_axis)

            # Check the orthogonalization and readjust the axes
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
                    self.dist_intershoulder = float(np.linalg.norm(self.shoulders[0] - self.shoulders[1]))

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

        colors = np.linspace(0, 1, len(handpoints_3D))  # Normalized values for color mapping

        # Update the scatter plot and colors
        scatter.set_offsets(handpoints_3D[:, :2])  # Update X et Y
        scatter.set_3d_properties(handpoints_3D[:, 2], "z")  # Update Z
        scatter.set_array(colors)  # Update colors
        scatter.set_clim(0, 1)  # Set color limits
        plt.draw()
        plt.pause(0.001)
