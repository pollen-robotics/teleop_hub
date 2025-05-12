import threading
import time
from collections import deque
from typing import Deque, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
from camera.camera import Camera  # type: ignore
from pyorbbecsdk import (  # type: ignore
    Config,
    Frame,
    OBAlignMode,
    OBFormat,
    OBSensorType,
    Pipeline,
)
from utils import load_config  # type: ignore


class Orbbec(Camera):
    """Orbbec camera class for depth and color stream."""

    def __init__(self, align_mode: str = "SW", enable_sync: bool = True) -> None:
        """Initialize the Orbbec camera.

        The frames are resized to the scale_percent defined in the config.yaml file,
        and they are captured by a thread and stored in a deque with a maximum length of 1

        Args:
            align_mode (str): Alignment mode, can be "HW" (hardware), "SW"(software), or "DISABLE".
            enable_sync (bool): Enable frame synchronization between color and depth streams.
        """
        super().__init__()

        # initialize the pipeline and config of the Orbbec camera
        self.config = Config()
        self.pipeline = Pipeline()

        self._set_pipeline()

        self._set_align_mode(align_mode, enable_sync)
        self.pipeline.start(self.config)

        self.depth_frame: Deque = deque(maxlen=1)

        # get the scale_percent from the config.yaml file
        self.scale_percent = load_config("config.yaml").get("scale_percent", 50)

        # parameters for colorizing depth image
        self.MIN_DEPTH = 0.05
        self.MAX_DEPTH = 3

        self.get_parameters()

        # start the thread to get frames
        self.frame_getter = threading.Thread(target=self.get_frame)
        self.frame_getter.start()

    def _set_pipeline(self) -> None:
        """Set the pipeline for the Orbbec camera."""
        color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = color_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.color_profile)

        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.depth_profile = depth_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.depth_profile)

    def _set_align_mode(self, align_mode: str, enable_sync: bool) -> None:
        """Set the alignment mode and enable frame synchronization.

        Args:
            align_mode (str): Alignment mode, can be "HW" (hardware), "SW"(software), or "DISABLE".
            enable_sync (bool): Enable frame synchronization between color and depth streams.
        """
        device = self.pipeline.get_device()
        device_info = device.get_device_info()
        device_pid = device_info.get_pid()

        if align_mode == "HW":
            if device_pid == 0x066B:
                self.config._set_align_mode(OBAlignMode.SW_MODE)
                print("Alignment mode : Software (auto for Femto Mega)")
            else:
                self.config._set_align_mode(OBAlignMode.HW_MODE)
                print("Alignment mode : Hardware")
        elif align_mode == "SW":
            self.config._set_align_mode(OBAlignMode.SW_MODE)
            print("Alignment mode : Software")
        else:
            self.config._set_align_mode(OBAlignMode.DISABLE)
            print("Alignment deactivated")

        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
                print("Frame synchronisation activated")
            except Exception as e:
                print(f"Synchronisation error: {e}")

    def get_parameters(self) -> None:
        """Get the camera parameters such as image shape, focal length, and intrisic matrix."""
        while True:
            frames = self.pipeline.wait_for_frames(500)

            if frames:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame and depth_frame:
                    self.original_image_shape = np.array(
                        [
                            color_frame.get_height(),
                            color_frame.get_width(),
                        ]
                    )
                    self.image_shape = (self.original_image_shape * (self.scale_percent / 100)).astype(int)
                    self.color_format = color_frame.get_format()

                    focal_length = self.image_shape[1]
                    center = (self.image_shape[1] / 2, self.image_shape[0] / 2)
                    self.camera_matrix = np.array(
                        [
                            [focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1],
                        ],
                        dtype="double",
                    )
                    break

    def get_frame(self) -> None:
        """Get the resized color (BGR) and depth frames from the camera and store them in the deques."""
        while True:
            frames = self.pipeline.wait_for_frames(500)
            if frames:
                # get the color frame
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_frame_bgr = self._frame_to_bgr_image(color_frame)
                    if color_frame_bgr is not None:
                        color_frame_resized = self._resize_frames(color_frame_bgr)
                        self.color_frame.append(color_frame_resized)

                # get the depth frame
                depth_frame = frames.get_depth_frame()
                depth_data = self._get_depth_data(depth_frame)
                if depth_data is not None:
                    self.depth_frame.append(depth_data)

    def _resize_frames(self, frame: np.ndarray) -> np.ndarray:
        """Resize the frames to the scale_percent defined in the config.yaml file.

        Args:
            frame (np.ndarray): The frame to be resized.

        Returns:
            np.ndarray: The resized frame.
        """
        new_dim = (self.image_shape[1], self.image_shape[0])
        return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

    def _frame_to_bgr_image(self, frame: Frame) -> Optional[np.ndarray]:
        """Convert the color frame to a BGR image.

        Args:
            frame (Frame): The frame to be converted.

        Returns:
            Optional[np.ndarray]: The converted BGR image.
        """
        data = np.asanyarray(frame.get_data())

        if self.color_format in [OBFormat.RGB, OBFormat.BGR]:
            image = data.reshape((self.original_image_shape[0], self.original_image_shape[1], 3))
            if self.color_format == OBFormat.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif self.color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            print(f"Unsupported color format: {self.color_format}")
            return None
        return image

    def _get_depth_data(self, depth_frame: Frame) -> Optional[np.ndarray]:
        """Convert and resize the depth frame to a numpy array.

        Args:
            depth_frame (Frame): The depth frame to be processed.

        Returns:
            Optional[np.ndarray]: The resized depth data.
        """
        if depth_frame and np.any(self.image_shape):
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(self.original_image_shape)
            depth_data_float = depth_data.astype(np.float32) * 10e-4  # convert to meters
            depth_data_resized = self._resize_frames(depth_data_float)
            return depth_data_resized
        return None

    def view_stream(self, show_color=True, show_depth=True) -> None:
        """Show the color and depth streams in separate windows.

        Args:
            show_color (bool): Show the color stream.
            show_depth (bool): Show the depth stream.
        """
        while True:
            try:
                color_data = self.color_frame[0]
                depth_data = self.depth_frame[0]

                if show_color and color_data is not None:
                    cv2.imshow("Color Viewer", color_data)

                if show_depth and depth_data is not None:
                    print(f"depth data max: {depth_data.max()}, min : {depth_data.min()}")
                    depth_image = np.clip(
                        (depth_data - self.MIN_DEPTH) / (self.MAX_DEPTH - self.MIN_DEPTH) * 255,
                        0,
                        255,
                    )
                    depth_colormap = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imshow("Depth Viewer", depth_colormap)

                key = cv2.waitKey(1)
                if key in [ord("q"), 27]:
                    break
            except KeyboardInterrupt:
                break

    def stop(self) -> None:
        """Stop the camera and close all OpenCV windows."""
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Orbbec stopped.")


if __name__ == "__main__":
    orbbec = Orbbec()
    while len(orbbec.color_frame) == 0:
        time.sleep(0.5)
    orbbec.view_stream()
    orbbec.stop()
