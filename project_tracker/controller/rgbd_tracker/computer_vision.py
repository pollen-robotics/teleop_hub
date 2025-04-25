import time
from typing import Optional, Tuple

import cv2  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
from camera.orbbec import Orbbec  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore

SHOULDER_CST = [11, 12]
ELBOWS_CST = [13, 14]
WRISTS_CST = [19, 20]
FACELANDMARKS_CST = [1, 152, 33, 263]  # Nose tip, Chin, Left eye left corner, Right eye right corner
HANDLANDMARKS_CST = [2, 4, 5, 8]  # thumb mcp, thumb tip, index mcp, index tip


class ComputerVision:
    def __init__(self, camera: Orbbec, up_mode: bool = True):
        """Initialise la classe ComputerVision."""
        self.camera = camera
        self.image_shape = camera.image_shape
        self.up_mode = up_mode
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
            smooth_landmarks=True,
            refine_face_landmarks=False,
        )
        self.initialize_landmarks()
        self.user_center = np.zeros(3)
        self.T_world_camera = np.eye(3)

        self.get_user_parameters()

        if self.up_mode:
            self.calibrate()

        print("Computer vision initialized")

    def initialize_landmarks(self):
        """Initialise les points clés pour les épaules, coudes, poignets, visage et mains."""
        self.shoulders = np.zeros((2, 3))
        self.elbows = np.zeros((2, 3))
        self.wrists = np.zeros((2, 3))
        self.face_points = np.zeros((len(FACELANDMARKS_CST), 3))
        self.left_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))

    def calibrate(self, nb_frames: int = 30):
        shoulders_tab = np.zeros((nb_frames, 2, 3))
        forehead_tab = np.zeros((nb_frames, 3))
        ite = 0

        while ite < nb_frames:
            try:
                color_frame, depth_frame = self.get_frames()
            except Exception:
                continue

            results = self.process_frame(color_frame)
            if not results:
                continue

            body_landmarks = self.get_2D_landmarks(results.pose_landmarks)
            face_landmarks = self.get_2D_landmarks(results.face_landmarks)
            if body_landmarks is not None and face_landmarks is not None:
                left_shoulder, right_shoulder = self.get_3D_landmarks_with_depth(
                    body_landmarks, depth_frame, SHOULDER_CST
                )
                forehead = self.get_3D_landmarks_with_depth(face_landmarks, depth_frame, [10])[0]

                if left_shoulder is not None and right_shoulder is not None and forehead is not None:
                    shoulders_tab[ite] = np.vstack((left_shoulder, right_shoulder))
                    forehead_tab[ite] = forehead
                    ite += 1

        shoulders_median = np.median(shoulders_tab, axis=0)
        forehead_median = np.median(forehead_tab, axis=0)

        angle = np.arctan(
            (forehead_median[2] - np.mean(shoulders_median[:, 2]))
            / (forehead_median[1] - (np.mean(shoulders_median[:, 1])))
        )

        print(f"Camera orientation angle: {np.degrees(angle):.2f}")
        self.T_world_camera = R.from_euler("x", -angle, degrees=False).as_matrix()

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Récupère les frames de couleur et de profondeur de la caméra."""
        color_frame = self.camera.color_frame[0]
        depth_frame = self.camera.depth_frame[0]
        return color_frame, depth_frame

    def process_frame(self, color_frame: np.ndarray) -> Optional[mp.solutions.holistic.Holistic]:
        """Traite une frame avec MediaPipe Holistic."""
        return self.holistic.process(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

    def get_2D_landmarks(self, original_landmarks) -> Optional[np.ndarray]:
        """Récupère les points clés 2D des résultats de MediaPipe."""
        if original_landmarks is None:
            return None
        original_landmarks = np.array([(lm.x, lm.y) for lm in original_landmarks.landmark])
        x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
        y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)
        x = np.clip(x_init, 0, self.image_shape[1] - 1)
        y = np.clip(y_init, 0, self.image_shape[0] - 1)
        return np.vstack((x, y)).T

    def get_3D_landmarks(self, original_landmarks) -> Optional[np.ndarray]:
        if original_landmarks is None:
            return None

        original_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in original_landmarks.landmark])
        x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
        y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)
        z = (original_landmarks[:, 2] * self.image_shape[1]).astype(int)

        x = np.clip(x_init, 0, self.image_shape[1] - 1)
        y = np.clip(y_init, 0, self.image_shape[0] - 1)
        return np.vstack((x, y, z)).T.astype(np.float32)

    def get_3D_landmarks_with_depth(self, landmarks: np.ndarray, depth_frame: np.ndarray, indices: list) -> np.ndarray:
        """Estime les points clés 3D avec la profondeur."""
        landmarks_3D = []
        for index in indices:
            landmark = landmarks[index]
            if landmark is not None:
                landmark_3D = self.estimate_3D_landmark_with_depth(landmark, depth_frame)
                landmarks_3D.append(landmark_3D)
            else:
                landmarks_3D.append(None)
        return np.array([lm for lm in landmarks_3D if lm is not None])

    def estimate_3D_landmark_with_depth(
        self, landmark: np.ndarray, depth_frame: np.ndarray, radius: int = 2
    ) -> Optional[np.ndarray]:
        """Estime un point clé 3D avec la profondeur."""
        if landmark is None:
            return None

        x, y = landmark[0], landmark[1]
        if (radius + 1 < x < self.camera.image_shape[1] - radius - 1) and (
            radius + 1 < y < self.camera.image_shape[0] - radius - 1
        ):
            z = np.median(depth_frame[int(y - radius) : int(y + radius), int(x - radius) : int(x + radius)])
        else:
            z = depth_frame[int(y), int(x)]

        z *= self.normalization_factor
        return np.vstack((x, y, z)).T.astype(np.float32)

    def get_user_parameters(self):
        """Récupère les paramètres de l'utilisateur."""
        print("Get user parameters")
        dist_intershoulder_list = []
        user_center_list = []
        nb_frames = 10

        while len(dist_intershoulder_list) < nb_frames:
            try:
                color_frame, depth_frame = self.get_frames()
            except Exception as e:
                print(f"Error getting frames: {e}")
                continue

            results = self.process_frame(color_frame)
            if not results:
                continue

            body_landmarks = self.get_2D_landmarks(results.pose_landmarks)
            if body_landmarks is not None:
                dist_intershoulder_list.append(
                    np.linalg.norm(body_landmarks[SHOULDER_CST[0]] - body_landmarks[SHOULDER_CST[1]])
                )

        self.dist_intershoulder = np.median(dist_intershoulder_list)
        self.normalization_factor = self.dist_intershoulder / 0.3

        while len(user_center_list) < nb_frames:
            try:
                color_frame, depth_frame = self.get_frames()
            except Exception:
                continue

            results = self.process_frame(color_frame)
            if not results:
                continue

            body_landmarks = self.get_2D_landmarks(results.pose_landmarks)
            if body_landmarks is not None:
                left_shoulder, right_shoulder = self.get_3D_landmarks_with_depth(
                    body_landmarks, depth_frame, SHOULDER_CST
                )
                if left_shoulder is not None and right_shoulder is not None:
                    user_center = (left_shoulder + right_shoulder) / 2.0
                    user_center_list.append(user_center)

        self.user_center = np.median(user_center_list, axis=0).reshape(3)

    def update_landmarks_coordinates(self, fixed_user: bool = True) -> bool:
        """Met à jour les coordonnées des points clés."""
        if len(self.camera.color_frame) == 0 or len(self.camera.depth_frame) == 0:
            return False

        color_frame, depth_frame = self.get_frames()
        results = self.process_frame(color_frame)
        if not results:
            return False

        body_landmarks = self.get_2D_landmarks(results.pose_landmarks)
        left_hand_landmarks = self.get_3D_landmarks(results.left_hand_landmarks)
        right_hand_landmarks = self.get_3D_landmarks(results.right_hand_landmarks)
        face_landmarks = self.get_3D_landmarks(results.face_landmarks)

        if body_landmarks is not None:
            if not fixed_user:
                self.update_shoulders(body_landmarks, depth_frame)
            self.update_elbows_and_wrists(body_landmarks, depth_frame)

        if left_hand_landmarks is not None and right_hand_landmarks is not None:
            self.update_hand_landmarks(left_hand_landmarks, right_hand_landmarks)
        if face_landmarks is not None:
            self.update_face_landmarks(face_landmarks, depth_frame)

        return True

    def update_shoulders(self, body_landmarks: np.ndarray, depth_frame: np.ndarray):
        """Met à jour les coordonnées des épaules."""
        left_shoulder, right_shoulder = self.get_3D_landmarks_with_depth(body_landmarks, depth_frame, SHOULDER_CST)
        if left_shoulder is not None:
            self.shoulders[0] = left_shoulder
        if right_shoulder is not None:
            self.shoulders[1] = right_shoulder

        if np.any(self.shoulders[0]) and np.any(self.shoulders[1]):
            self.user_center = (self.shoulders[0] + self.shoulders[1]) / 2.0
            self.dist_intershoulder = float(np.linalg.norm(self.shoulders[0] - self.shoulders[1]))

    def update_elbows_and_wrists(self, body_landmarks: np.ndarray, depth_frame: np.ndarray):
        """Met à jour les coordonnées des coudes et des poignets."""
        for i in range(2):
            elbow = self.estimate_3D_landmark_with_depth(body_landmarks[ELBOWS_CST[i]], depth_frame)
            if elbow is not None:
                self.elbows[i] = elbow
            wrist = self.estimate_3D_landmark_with_depth(body_landmarks[WRISTS_CST[i]], depth_frame, radius=10)
            if wrist is not None:
                self.wrists[i] = wrist

    def update_hand_landmarks(self, left_hand_landmarks: np.ndarray, right_hand_landmarks: np.ndarray):
        """Met à jour les coordonnées des points clés des mains."""
        for hand_landmarks, points in zip(
            [left_hand_landmarks, right_hand_landmarks], [self.left_hand_points, self.right_hand_points]
        ):
            if hand_landmarks is not None:
                for i, lm_cst in enumerate(HANDLANDMARKS_CST):
                    finger_coord = hand_landmarks[lm_cst]
                    if finger_coord is not None:
                        points[i] = finger_coord

    def update_face_landmarks(self, face_landmarks: np.ndarray, depth_frame: np.ndarray):
        """Met à jour les coordonnées des points clés du visage."""
        if face_landmarks is not None:
            for i, lm_cst in enumerate(FACELANDMARKS_CST):
                face_coord = face_landmarks[lm_cst]
                if face_coord is not None:
                    self.face_points[i] = face_coord

    def visualization_landmarks(
        self,
        color_frame: np.ndarray,
        left_goal_pose: np.ndarray,
        right_goal_pose: np.ndarray,
        roll: float,
        pitch: float,
        yaw: float,
        text_on: bool = True,
        body_on: bool = True,
        face_on: bool = True,
    ) -> np.ndarray:
        """Visualise les points clés sur l'image."""
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
        return color_frame

    def init_plot(self):
        """Initialise le plot 3D."""
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
        """Met à jour le plot 3D."""
        handpoints_3D = self.left_hand_points_3D
        colors = np.linspace(0, 1, len(handpoints_3D))  # Normalized values for color mapping
        scatter.set_offsets(handpoints_3D[:, :2])  # Update X et Y
        scatter.set_3d_properties(handpoints_3D[:, 2], "z")  # Update Z
        scatter.set_array(colors)  # Update colors
        scatter.set_clim(0, 1)  # Set color limits
        plt.draw()
        plt.pause(0.001)

    def stop(self):
        """Arrête la capture vidéo et ferme MediaPipe."""
        self.holistic.close()
        self.camera.stop()


if __name__ == "__main__":
    camera = Orbbec()
    computer_vision = ComputerVision(camera)

    time.sleep(3)
    while True:
        if not computer_vision.update_landmarks_coordinates():
            print("No landmarks available.")
        else:
            color_frame = computer_vision.camera.color_frame[0]
            frame = computer_vision.visualization_landmarks(color_frame, np.eye(4), np.eye(4), 0, 0, 0, text_on=False)
            cv2.imshow("Color Viewer", frame)
            cv2.waitKey(1)
