import threading
import time
from typing import Optional

import numpy as np
from camera.camera import Camera  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from utils import load_config  # type: ignore


class RGBCamera(Camera):
    """RGB camera class for color stream.

    This class is used to get the color stream from a camera.
    The frames are captured by a thread and stored in a deque with a maximum length of 1.
    It can be used with a USB camera or an IP camera, and the frames are resized.
    All the parameters are obtained from the config.yaml file.
    """

    def __init__(self) -> None:
        """Initialize the camera stream."""
        super().__init__()

        # Get the camera parameters from the config file
        config = load_config("config.yaml")

        usb_mode = config.get("camera", {}).get("usb_mode", True)
        camera_id = config.get("camera", {}).get("camera_id", "0")
        with_calibration = config.get("camera", {}).get("with_calibration", False)

        # get the capture device
        if usb_mode:
            camera_index = int(camera_id) if camera_id else 0
            self.cap = self.cv2.VideoCapture(camera_index)
        else:
            self.cap = self.cv2.VideoCapture(f"http://{camera_id}:8080/video")

        # Start the thread to get the camera frame
        self.frame_getter = threading.Thread(target=self.get_frame, daemon=True)
        self.frame_getter.start()

        # Camera calibration to get parameters
        self.calibrate_camera(with_calibration, config)

    def calibrate_camera(self, with_calibration: bool, config: dict) -> None:
        """Get the camera parameters.

        Args:
            with_calibration (bool): If True, use the parameters given by the camera calibration script.
                If False, use manual camera parameters.
            config (dict): The configuration dictionary, got from the config.yaml file.
        """
        if with_calibration:
            camera_config = config.get("camera", {})
            self.camera_matrix = np.array(camera_config["camera_matrix"], dtype=np.float32)
            self.dist_coeffs = np.array(camera_config["dist_coeffs"], dtype=np.float32)

        else:
            while len(self.color_frame) == 0:
                time.sleep(0.05)
            frame_shape = self.color_frame[0].shape
            focal_length = frame_shape[1]
            center = (frame_shape[1] / 2, frame_shape[0] / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype="double",
            )
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def get_frame(self) -> None:
        """Get the camera frame with post-processing.

        This function runs in a separate thread to avoid blocking the main thread,
        and replaces the previous frame with the new one in the deque.
        """
        while True:
            success, frame = self.cap.read()
            if success:
                frame_processed = self._post_process_image(frame)
                self.color_frame.append(frame_processed)
            time.sleep(0.005)

    def _post_process_image(self, frame: np.ndarray) -> np.ndarray:
        """Post-process the image.

        This function is used to enhance the image for better cube detection.
        It applies a Gaussian blur and Canny edge detection to the image.

        Args:
            frame: The image frame to process.
        Returns:
            The processed image.
        """
        frame_gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)

        blurred = self.cv2.GaussianBlur(frame_gray, (5, 5), 1)
        edges = self.cv2.Canny(blurred, threshold1=50, threshold2=150)

        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = self.cv2.dilate(edges, kernel, iterations=1)
        edges_weight = self.cv2.normalize(edges_dilated, None, 0, 1, self.cv2.NORM_MINMAX)

        non_edge_weight = 1 - edges_weight
        blurred_image = self.cv2.GaussianBlur(frame_gray, (9, 9), 3)

        enhanced_image = edges_weight * frame_gray + non_edge_weight * blurred_image
        enhanced_image = self.cv2.normalize(enhanced_image, None, 0, 255, self.cv2.NORM_MINMAX)

        return enhanced_image

    def get_frame_with_cube_pose(self, cube_pose_list: list) -> Optional[np.ndarray]:
        """Show information about the cubes poses on the frame.

        This function draws the cube poses on the frame and displays their position and orientation.
        It uses the camera parameters to draw the axes of the cube poses.
        It also shows the roll, pitch, and yaw angles of the cube poses.

        Args:
            cube_pose_list: List of cube poses (left and right)
                Each pose is a 4x4 matrix representing the cube's position and orientation.
        Returns:
            The frame with the cube infos.
        """
        if len(self.color_frame) == 0:
            return None

        frame = self.color_frame[0]
        for i, cube_pose in enumerate(cube_pose_list):
            if cube_pose is not None:
                side = "right" if i == 1 else "left"
                rvec = R.from_matrix(cube_pose[:3, :3]).as_rotvec()
                tvec = cube_pose[:3, 3]
                self.cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

                self.cv2.putText(
                    frame,
                    f"{side} cube - x: {np.round(tvec[0],3)}, y: {np.round(tvec[1],3)}, z: {np.round(tvec[2],3)}",
                    (10, 20 + 40 * i),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )

                roll, pitch, yaw = R.from_matrix(cube_pose[:3, :3]).as_euler("xyz", degrees=True)

                self.cv2.putText(
                    frame,
                    f"{side} cube - roll: {np.round(roll)}, pitch: {np.round(pitch)}, yaw: {np.round(yaw)}",
                    (10, 40 + 40 * i),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )

        return frame

    def get_frame_with_markers(self, markers_dict: dict) -> np.ndarray:
        """Show information about the markers poses on the frame.

        This function draws the markers poses on the frame and displays their position and orientation.
        It uses the camera parameters to draw the axes of the markers poses.

        Args:
            markers_dict: Dictionary of markers poses.
                Each pose is a dictionary containing the rotation and translation vectors.
                The keys are the marker IDs and the values are dictionaries with the keys 'rvec' and 'tvec'.
        Returns:
            The frame with the markers infos.
        """
        frame = self.color_frame[0]
        for marker_id, data in markers_dict.items():
            rvec = data["rvec"]
            tvec = data["tvec"]

            self.cv2.drawFrameAxes(
                frame,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                0.03,
            )
            trans = tvec.flatten()
            self.cv2.putText(
                frame,
                f"{marker_id} - {np.round(trans,3)}",
                (10, 20 + 20 * marker_id),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
        # self.cv2.aruco.drawDetectedMarkers(frame, [corners], np.array([[marker_id]]))
        return frame
