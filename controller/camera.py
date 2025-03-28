import threading
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros((5, 1))

        self.frame = deque(maxlen=1)

        self.frame_getter = threading.Thread(target=self.get_frame, daemon=True)
        self.frame_getter.start()

        self.calibrate_camera()

    def calibrate_camera(self):
        while len(self.frame) == 0:
            time.sleep(0.05)
        frame = self.frame[0]
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def get_frame(self):
        while True:
            success, frame = self.cap.read()
            if success:
                frame_processed = self.post_process_image(frame)
                self.frame.append(frame_processed)
            time.sleep(0.05)

    def post_process_image(self, frame):
        frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # add CLAHE to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_clahe = clahe.apply(frame_processed)
        return frame_clahe

    def get_frame_with_cube_pose(self, cube_pose_list):
        if len(self.frame) == 0:
            return

        frame = self.frame[0]
        for i, cube_pose in enumerate(cube_pose_list):
            if cube_pose is not None:
                side = "right" if i == 1 else "left"
                rvec = R.from_matrix(cube_pose[:3, :3]).as_rotvec()
                tvec = cube_pose[:3, 3]
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

                cv2.putText(
                    frame,
                    f"{side} cube - x: {np.round(tvec[0],3)}, y: {np.round(tvec[1],3)}, z: {np.round(tvec[2],3)}",
                    (10, 20 + 40 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )

                roll, pitch, yaw = R.from_matrix(cube_pose[:3, :3]).as_euler("xyz", degrees=True)

                cv2.putText(
                    frame,
                    f"{side} cube - roll: {np.round(roll)}, pitch: {np.round(pitch)}, yaw: {np.round(yaw)}",
                    (10, 40 + 40 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )

        return frame
