import collections
import threading
import time
from typing import Optional

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
    def __init__(self, align_mode="SW", enable_sync=True):
        self.scale_percent = load_config("config.yaml").get("scale_percent", 50)
        self.config = Config()
        self.pipeline = Pipeline()

        color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = color_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.color_profile)

        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.depth_profile = depth_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.depth_profile)

        self.MIN_DEPTH = 0.05
        self.MAX_DEPTH = 3

        self.set_align_mode(align_mode, enable_sync)

        self.pipeline.start(self.config)

        self.color_frame = collections.deque(maxlen=1)
        self.depth_frame = collections.deque(maxlen=1)

        self.original_image_shape = np.zeros(2, dtype=np.int32)
        self.image_shape = np.zeros(2, dtype=np.int32)

        self.color_format = None

        self.get_parameters()

        self.frame_getter = threading.Thread(target=self.get_frame)
        self.frame_getter.start()

    def set_align_mode(self, align_mode, enable_sync):
        device = self.pipeline.get_device()
        device_info = device.get_device_info()
        device_pid = device_info.get_pid()

        if align_mode == "HW":
            if device_pid == 0x066B:
                self.config.set_align_mode(OBAlignMode.SW_MODE)
                print("Alignment mode : Software (auto for Femto Mega)")
            else:
                self.config.set_align_mode(OBAlignMode.HW_MODE)
                print("Alignment mode : Hardware")
        elif align_mode == "SW":
            self.config.set_align_mode(OBAlignMode.SW_MODE)
            print("Alignment mode : Software")
        else:
            self.config.set_align_mode(OBAlignMode.DISABLE)
            print("Alignment deactivated")

        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
                print("Frame synchronisation activated")
            except Exception as e:
                print(f"Cynchronisation error: {e}")

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
                    self.original_image_shape[:] = [color_frame.get_height(), color_frame.get_width()]
                    self.image_shape = (self.original_image_shape * (self.scale_percent / 100)).astype(int)
                    self.color_format = color_frame.get_format()

                    focal_length = self.image_shape[1]
                    center = (self.image_shape[1] / 2, self.image_shape[0] / 2)
                    self.camera_matrix = np.array(
                        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                        dtype="double",
                    )
                    break

    def resize_frames(self, frame):
        new_dim = (self.image_shape[1], self.image_shape[0])
        return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

    def get_frame(self):
        while True:
            frames = self.pipeline.wait_for_frames(500)
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_frame_bgr = self.frame_to_bgr_image(color_frame)
                    color_frame_resized = self.resize_frames(color_frame_bgr)
                    self.color_frame.append(color_frame_resized)

                depth_frame = frames.get_depth_frame()
                depth_data = self.get_depth_data(depth_frame)
                if depth_data is not None:
                    self.depth_frame.append(depth_data)

    def get_depth_data(self, depth_frame) -> Optional[Frame]:
        if depth_frame and np.any(self.image_shape):
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(self.original_image_shape)
            depth_data_float = depth_data.astype(np.float32) * 10e-4
            depth_data_resized = self.resize_frames(depth_data_float)
            return depth_data_resized
        return None

    def view_stream(self, show_color=True, show_depth=True):
        while True:
            try:
                color_data = self.color_frame[0]
                depth_data = self.depth_frame[0]

                if show_color and color_data is not None:
                    cv2.imshow("Color Viewer", color_data)

                if show_depth and depth_data is not None:
                    print(f"depth data max: {depth_data.max()}, min : {depth_data.min()}")
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
            image = data.reshape((self.original_image_shape[0], self.original_image_shape[1], 3))
            if self.color_format == OBFormat.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif self.color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            print(f"Unsupported color format: {self.color_format}")
            return None
        return image


if __name__ == "__main__":
    orbbec = Orbbec()
    while len(orbbec.color_frame) == 0:
        time.sleep(0.5)
    orbbec.view_stream()
    orbbec.stop()
