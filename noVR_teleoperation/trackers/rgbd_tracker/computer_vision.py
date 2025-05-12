import time
from typing import Optional, Tuple

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
from camera.camera import Camera  # type: ignore
from camera.orbbec import Orbbec  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore

# Constants for the landmark indices, based on the MediaPipe Holistic model
# Face : Nose tip, Chin, Left eye left corner, Right eye right corner
# Hand : Thumb mcp, thumb tip, index mcp, index tip
# ---------------------------------------------------------------------------
SHOULDER_CST = [11, 12]
ELBOWS_CST = [13, 14]
WRISTS_CST = [19, 20]
FACELANDMARKS_CST = [1, 152, 33, 263]
HANDLANDMARKS_CST = [2, 4, 5, 8]
# ---------------------------------------------------------------------------


class ComputerVision:
    """Class for computer vision using MediaPipe."""

    def __init__(self, camera: Camera, up_mode: bool = True) -> None:
        """Initialize the computer vision class.

        Args:
            camera (Camera): The camera object.
            up_mode (bool): If True, calibrate the camera with the angle between the shoulders and the forehead.
                This is used to align the camera with the user.
        """
        self.camera = camera
        self.image_shape = self.camera.image_shape
        self.up_mode = up_mode

        # Initialize MediaPipe Holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
            smooth_landmarks=True,
            refine_face_landmarks=False,
        )

        self._initialize_landmarks()
        self.user_center = np.zeros(3)
        self.T_world_camera = np.eye(3)

        self._get_user_parameters()

        if self.up_mode:
            self.calibrate()

        print("Computer vision initialized")

    def _initialize_landmarks(self) -> None:
        """Initialize the keypoints (shoulders, elbows, wrists, face and hands)."""
        self.shoulders = np.zeros((2, 3))
        self.elbows = np.zeros((2, 3))
        self.wrists = np.zeros((2, 3))
        self.face_points = np.zeros((len(FACELANDMARKS_CST), 3))
        self.left_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))
        self.right_hand_points = np.zeros((len(HANDLANDMARKS_CST), 3))

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the color and depth frames from the camera.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The color and depth frames.
        """
        color_frame = self.camera.color_frame[0]
        depth_frame = self.camera.depth_frame[0]
        return color_frame, depth_frame

    def process_frame(self, color_frame: np.ndarray) -> Optional[mp.solutions.holistic.Holistic]:
        """Process the color frame with MediaPipe Holistic.

        Returns:
            Optional[mp.solutions.holistic.Holistic]: The results of the MediaPipe Holistic processing.
        """
        return self.holistic.process(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

    def _get_user_parameters(self) -> None:
        """Get the user parameters (shoulder distance and user center)."""
        dist_intershoulder_list: list = []
        user_center_list: list = []
        nb_frames = 10

        # Get the distance between the shoulders as the median of the last 10 frames
        while len(dist_intershoulder_list) < nb_frames:
            try:
                color_frame, depth_frame = self.get_frames()
            except Exception as e:
                print(f"Error getting frames: {e}")
                continue

            results = self.process_frame(color_frame)
            if not results:
                continue

            body_landmarks = self._get_2D_landmarks(results.pose_landmarks)
            if body_landmarks is not None:
                dist_intershoulder_list.append(
                    np.linalg.norm(body_landmarks[SHOULDER_CST[0]] - body_landmarks[SHOULDER_CST[1]])
                )

        self.dist_intershoulder = np.median(dist_intershoulder_list)
        self.normalization_factor = self.dist_intershoulder / 0.3

        # then, get the user center as the median of the last 10 frames
        while len(user_center_list) < nb_frames:
            try:
                color_frame, depth_frame = self.get_frames()
            except Exception:
                continue

            results = self.process_frame(color_frame)
            if not results:
                continue

            body_landmarks = self._get_2D_landmarks(results.pose_landmarks)
            if body_landmarks is not None:
                left_shoulder, right_shoulder = self._get_3D_landmarks_with_depth(
                    body_landmarks, depth_frame, SHOULDER_CST
                )
                if left_shoulder is not None and right_shoulder is not None:
                    user_center = (left_shoulder + right_shoulder) / 2.0
                    user_center_list.append(user_center)

        self.user_center = np.median(user_center_list, axis=0).reshape(3)

    def calibrate(self, nb_frames: int = 30) -> None:
        """Calibrate the camera to get the angle between the shoulders and the forehead.

        This is used to align the camera with the user.

        Args:
            nb_frames (int): Number of frames to use for calibration.
        """
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

            body_landmarks = self._get_2D_landmarks(results.pose_landmarks)
            face_landmarks = self._get_2D_landmarks(results.face_landmarks)
            if body_landmarks is not None and face_landmarks is not None:
                left_shoulder, right_shoulder = self._get_3D_landmarks_with_depth(
                    body_landmarks, depth_frame, SHOULDER_CST
                )
                forehead = self._get_3D_landmarks_with_depth(face_landmarks, depth_frame, [10])[0]

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

    def update_landmarks_coordinates(self, fixed_user: bool = True) -> bool:
        """Update the coordinates of the landmarks.

        Args:
            fixed_user (bool): If True, the user is fixed and the coordinates are not updated.
        Returns:
            bool: True if the coordinates are updated, False otherwise."""
        if len(self.camera.color_frame) == 0 or len(self.camera.depth_frame) == 0:
            return False

        color_frame, depth_frame = self.get_frames()
        results = self.process_frame(color_frame)
        if not results:
            return False

        body_landmarks = self._get_2D_landmarks(results.pose_landmarks)
        left_hand_landmarks = self._get_3D_landmarks(results.left_hand_landmarks)
        right_hand_landmarks = self._get_3D_landmarks(results.right_hand_landmarks)
        face_landmarks = self._get_3D_landmarks(results.face_landmarks)

        if body_landmarks is not None:
            if not fixed_user:
                self._update_shoulders(body_landmarks, depth_frame)
            self._update_elbows_and_wrists(body_landmarks, depth_frame)

        if left_hand_landmarks is not None and right_hand_landmarks is not None:
            self._update_hand_landmarks(left_hand_landmarks, right_hand_landmarks)
        if face_landmarks is not None:
            self._update_face_landmarks(face_landmarks)

        return True

    def _get_2D_landmarks(self, original_landmarks) -> Optional[np.ndarray]:
        """Get the 2D landmarks from the original landmarks.

        Args:
            original_landmarks: The original landmarks from MediaPipe.

        Returns:
            Optional[np.ndarray]: The 2D landmarks in the image coordinates.
        """
        if original_landmarks is None:
            return None
        original_landmarks = np.array([(lm.x, lm.y) for lm in original_landmarks.landmark])
        x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
        y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)
        x = np.clip(x_init, 0, self.image_shape[1] - 1)
        y = np.clip(y_init, 0, self.image_shape[0] - 1)
        return np.vstack((x, y)).T

    def _get_3D_landmarks(self, original_landmarks) -> Optional[np.ndarray]:
        """Get the 3D landmarks from the original landmarks.

        Args:
            original_landmarks: The original landmarks from MediaPipe.
        Returns:
            Optional[np.ndarray]: The 3D landmarks in the image coordinates.
        """
        if original_landmarks is None:
            return None

        original_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in original_landmarks.landmark])
        x_init = (original_landmarks[:, 0] * self.image_shape[1]).astype(int)
        y_init = (original_landmarks[:, 1] * self.image_shape[0]).astype(int)
        z = (original_landmarks[:, 2] * self.image_shape[1]).astype(int)

        x = np.clip(x_init, 0, self.image_shape[1] - 1)
        y = np.clip(y_init, 0, self.image_shape[0] - 1)
        return np.vstack((x, y, z)).T.astype(np.float32)

    def _get_3D_landmarks_with_depth(
        self, landmarks: np.ndarray, depth_frame: np.ndarray, indices: list
    ) -> np.ndarray:
        """Estimate the 3D coordinates of landmarks from the 2D landmarks and the depth data.

        Args:
            landmarks (np.ndarray): The 2D landmarks in the image coordinates.
            depth_frame (np.ndarray): The depth frame from the camera.
            indices (list): The indices of the landmarks to estimate.
        Returns:
            np.ndarray: The 3D coordinates of the specified landmarks in the image coordinates.
        """
        landmarks_3D = []
        for index in indices:
            landmark = landmarks[index]
            if landmark is not None:
                landmark_3D = self._estimate_3D_landmark_with_depth(landmark, depth_frame)
                landmarks_3D.append(landmark_3D)
            else:
                landmarks_3D.append(None)
        return np.array([lm for lm in landmarks_3D if lm is not None])

    def _estimate_3D_landmark_with_depth(
        self, landmark: np.ndarray, depth_frame: np.ndarray, radius: int = 2
    ) -> Optional[np.ndarray]:
        """Estimate the 3D coordinates of a specific landmark.
        Args:
            landmark (np.ndarray): The 2D landmark in the image coordinates.
            depth_frame (np.ndarray): The depth frame from the camera.
            radius (int): The radius around the landmark to use for depth estimation.

        Returns:
            np.ndarray: The 3D coordinates of the landmark in the image coordinates.
        """
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

    def _update_shoulders(self, body_landmarks: np.ndarray, depth_frame: np.ndarray) -> None:
        """Update the coordinates of the shoulders and the user_center.

        (Only used if not fixed_user mode, not available yet).

        Args:
            body_landmarks (np.ndarray): The 2D landmarks in the image coordinates.
            depth_frame (np.ndarray): The depth frame from the camera.
        """
        left_shoulder, right_shoulder = self._get_3D_landmarks_with_depth(body_landmarks, depth_frame, SHOULDER_CST)
        if left_shoulder is not None:
            self.shoulders[0] = left_shoulder
        if right_shoulder is not None:
            self.shoulders[1] = right_shoulder

        if np.any(self.shoulders[0]) and np.any(self.shoulders[1]):
            self.user_center = (self.shoulders[0] + self.shoulders[1]) / 2.0
            self.dist_intershoulder = float(np.linalg.norm(self.shoulders[0] - self.shoulders[1]))

    def _update_elbows_and_wrists(self, body_landmarks: np.ndarray, depth_frame: np.ndarray) -> None:
        """Update the coordinates of the elbows and wrists.

        Args:
            body_landmarks (np.ndarray): The 2D landmarks in the image coordinates.
            depth_frame (np.ndarray): The depth frame from the camera.
        """
        for i in range(2):
            elbow = self._estimate_3D_landmark_with_depth(body_landmarks[ELBOWS_CST[i]], depth_frame)
            if elbow is not None:
                self.elbows[i] = elbow
            wrist = self._estimate_3D_landmark_with_depth(body_landmarks[WRISTS_CST[i]], depth_frame, radius=10)
            if wrist is not None:
                self.wrists[i] = wrist

    def _update_hand_landmarks(self, left_hand_landmarks: np.ndarray, right_hand_landmarks: np.ndarray) -> None:
        """Update the coordinates of the hand landmarks.

        Args:
            left_hand_landmarks (np.ndarray): The 3D landmarks of the left hand.
            right_hand_landmarks (np.ndarray): The 3D landmarks of the right hand.
        """
        for hand_landmarks, points in zip(
            [left_hand_landmarks, right_hand_landmarks],
            [self.left_hand_points, self.right_hand_points],
        ):
            if hand_landmarks is not None:
                for i, lm_cst in enumerate(HANDLANDMARKS_CST):
                    finger_coord = hand_landmarks[lm_cst]
                    if finger_coord is not None:
                        points[i] = finger_coord

    def _update_face_landmarks(self, face_landmarks: np.ndarray) -> None:
        """Update the coordinates of the face landmarks.

        Args:
            face_landmarks (np.ndarray): The 3D landmarks of the face.
        """
        if face_landmarks is not None:
            for i, lm_cst in enumerate(FACELANDMARKS_CST):
                face_coord = face_landmarks[lm_cst]
                if face_coord is not None:
                    self.face_points[i] = face_coord

    def visualization_landmarks(
        self,
        color_frame: np.ndarray,
        left_goal_pose: Optional[np.ndarray],
        right_goal_pose: Optional[np.ndarray],
        roll: float,
        pitch: float,
        yaw: float,
        text_on: bool = True,
        body_on: bool = True,
        face_on: bool = True,
    ) -> np.ndarray:
        """Visualize the landmarks on the color frame.

        Args:
            color_frame (np.ndarray): The color frame.
            left_goal_pose (np.ndarray): The pose of the left goal.
            right_goal_pose (np.ndarray): The pose of the right goal.
            roll (float): The roll angle of the face.
            pitch (float): The pitch angle of the face.
            yaw (float): The yaw angle of the face.
            text_on (bool): If True, display the text on the frame.
            body_on (bool): If True, display the body landmarks on the frame.
            face_on (bool): If True, display the face landmarks on the frame.
        Returns:
            np.ndarray: The color frame with the landmarks drawn on it.
        """
        cv2.circle(
            color_frame,
            (int(self.user_center[0]), int(self.user_center[1])),
            5,
            (255, 255, 255),
            -1,
        )
        if text_on:
            if left_goal_pose is not None:
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
            if right_goal_pose is not None:
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
                cv2.circle(
                    color_frame,
                    (int(self.wrists[i][0]), int(self.wrists[i][1])),
                    5,
                    (0, 255, 0),
                    -1,
                )
                cv2.circle(
                    color_frame,
                    (int(self.elbows[i][0]), int(self.elbows[i][1])),
                    5,
                    (255, 0, 0),
                    -1,
                )
                cv2.circle(
                    color_frame,
                    (int(self.shoulders[i][0]), int(self.shoulders[i][1])),
                    5,
                    (0, 0, 255),
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
        return color_frame

    def stop(self):
        """Stop the camera and close the MediaPipe Holistic."""
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
