import json
import os
import threading
import time
from collections import deque
from typing import Deque, Optional

import cv2  # type: ignore
import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


class Camera:
    def __init__(self, phone_mode: bool = False, camera_ip: Optional[str] = None, with_calibration: bool = False):
        if phone_mode:
            camera_ip = '172.16.0.56'
            self.cap = cv2.VideoCapture(f'http://{camera_ip}:8080/video')
        else:
            camera_ip = 'webcam'
            self.cap = cv2.VideoCapture(0)

        self.frame: Deque[np.ndarray] = deque(maxlen=1)
        self.frame_getter = threading.Thread(target=self.get_frame, daemon=True)
        self.frame_getter.start()

        self.calibrate_camera(with_calibration, camera_ip)

    def calibrate_camera(self, with_calibration: bool, camera_ip: str):
        if with_calibration:
            current_dir = os.path.dirname(__file__)
            json_path = os.path.join(current_dir, '..', 'cameras_param', f'{camera_ip}.json')
            json_path = os.path.abspath(json_path)

            with open(json_path, 'r') as f:
                camera_params = json.load(f)

                self.camera_matrix = np.array(camera_params['camera_matrix'], dtype=np.float32)
                self.dist_coeffs = np.array(camera_params['dist_coeffs'], dtype=np.float32)

        else:
            while len(self.frame) == 0:
                time.sleep(0.05)
            frame = self.frame[0]
            focal_length = frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype="double",
            )
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def get_frame(self):
        while True:
            success, frame = self.cap.read()
            if success:
                frame_processed = self.post_process_image(frame)
                self.frame.append(frame_processed)
            time.sleep(0.005)

    def post_process_image(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 1)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        edges_weight = cv2.normalize(edges_dilated, None, 0, 1, cv2.NORM_MINMAX)

        non_edge_weight = 1 - edges_weight
        blurred_image = cv2.GaussianBlur(frame_gray, (9, 9), 3)

        enhanced_image = edges_weight * frame_gray + non_edge_weight * blurred_image
        enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)

        return enhanced_image

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

    def get_frame_with_markers(self, markers_dict):
        frame = self.frame[0]
        for marker_id, data in markers_dict.items():
            rvec = data["rvec"]
            tvec = data["tvec"]

            cv2.drawFrameAxes(frame, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.03)
            trans = tvec.flatten()
            cv2.putText(
                frame,
                f"{marker_id} - {np.round(trans,3)}",
                (10, 20 + 20 * marker_id),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
        # cv2.aruco.drawDetectedMarkers(frame, [corners], np.array([[marker_id]]))
        return frame
