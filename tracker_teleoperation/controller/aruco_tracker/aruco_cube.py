import time
from typing import Optional

import cv2  # type: ignore
import cv2.aruco as aruco  # type: ignore
import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore

from tracker_teleoperation.controller.aruco_tracker.camera import Camera
from tracker_teleoperation.controller.tracker import (  # type: ignore
    Tracker,
    TrackerType,
)


class ArucoCube(Tracker):
    """Class to detect and track an ArUco cube."""

    def __init__(
        self, arm: str, camera: Camera, marker_size: float = 0.05
    ) -> None:
        """Initialize the ArUco cube.

        Args:
            camera (Camera): Camera object to get the frame.
            marker_size (float): Size of the markers in meters.
            arm (str): Corresponding arm for teleoperation. Either "l_arm" or "r_arm".
        """
        super().__init__(arm)
        self.tracker_type = TrackerType.ARUCO

        self.camera = camera
        self.frame: Optional[np.ndarray] = None

        self.marker_size = marker_size

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        aruco_param = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco_param)

        self.cube = None
        self.define_cube()

        self.markers_dict: dict = {}

    def define_cube(self) -> None:
        """Define the cube with specific markers, depending on the arm."""
        c_pt = self.marker_size / 2

        if self.arm == "r_arm":
            back_marker = 0
            up_marker = 1
            left_marker = 2
            down_marker = 3
            right_marker = 4
            front_marker = 5

        else:
            back_marker = 6
            up_marker = 7
            left_marker = 8
            down_marker = 11
            right_marker = 9
            front_marker = 10

        self.cube_ids = np.array(
            [
                back_marker,
                up_marker,
                left_marker,
                down_marker,
                right_marker,
                front_marker,
            ]
        )

        self.cube_corners = [
            np.array(
                [
                    [-c_pt, c_pt, c_pt],
                    [c_pt, c_pt, c_pt],
                    [c_pt, -c_pt, c_pt],
                    [-c_pt, -c_pt, c_pt],
                ],
                dtype=np.float32,
            ),  # back face (left corner at the right bottom)
            np.array(
                [
                    [-c_pt, -c_pt, c_pt],
                    [c_pt, -c_pt, c_pt],
                    [c_pt, -c_pt, -c_pt],
                    [-c_pt, -c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),  # up face
            np.array(
                [
                    [-c_pt, c_pt, -c_pt],
                    [-c_pt, c_pt, c_pt],
                    [-c_pt, -c_pt, c_pt],
                    [-c_pt, -c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),  # left face
            np.array(
                [
                    [-c_pt, c_pt, -c_pt],
                    [c_pt, c_pt, -c_pt],
                    [c_pt, c_pt, c_pt],
                    [-c_pt, c_pt, c_pt],
                ],
                dtype=np.float32,
            ),  # down face
            np.array(
                [
                    [c_pt, c_pt, c_pt],
                    [c_pt, c_pt, -c_pt],
                    [c_pt, -c_pt, -c_pt],
                    [c_pt, -c_pt, c_pt],
                ],
                dtype=np.float32,
            ),  # right face
            np.array(
                [
                    [-c_pt, -c_pt, -c_pt],
                    [c_pt, -c_pt, -c_pt],
                    [c_pt, c_pt, -c_pt],
                    [-c_pt, c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),  # front face
        ]  # 0,1,2,3,4,5

        self.cube = aruco.Board(self.cube_corners, self.aruco_dict, self.cube_ids)

    def update_tracker_pose(self) -> Optional[np.ndarray]:
        """Update the cube pose using the camera frame.

        This function detects the markers in the frame and estimates the pose of the cube.

        Returns:
            tracker_pose (np.ndarray): The pose of the cube in the camera frame.
        """
        try:
            self.frame = self.camera.frame[0]
        except IndexError:
            time.sleep(0.01)
            print("No frame")
            return None

        markers_corners, markers_ids = self.detect_markers()

        if markers_ids is not None and len(markers_ids) > 0:
            markers_ids = np.array(markers_ids, dtype=np.int32)

            _, rvec, tvec = cv2.aruco.estimatePoseBoard(
                markers_corners,
                markers_ids,
                self.cube,
                self.camera.camera_matrix,
                self.camera.dist_coeffs,
                None,
                None,
            )

            if rvec is not None and tvec is not None:
                tracker_pose = np.eye(4)
                tracker_pose[:3, :3] = R.from_rotvec(rvec.reshape(1, 3)).as_matrix()

                tracker_pose[:3, 3] = tvec.flatten()
                self.tracker_pose = tracker_pose

        return self.tracker_pose

    def detect_markers(self) -> tuple[np.ndarray, np.ndarray]:
        """Detect markers in the current frame.

        Returns:
            marker_corners (np.ndarray): The corners of the detected markers.
            marker_ids (np.ndarray): The IDs of the detected markers.
        """
        marker_corners, marker_ids, _ = self.detector.detectMarkers(self.frame)
        return marker_corners, marker_ids

    def estimate_PoseSingleMarkers(
        self, corners: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the pose of single markers.

        Args:
            corners (np.ndarray): The corners of the detected markers.

        Returns:
            rvecs (np.ndarray): The rotation vectors of the markers.
            tvecs (np.ndarray): The translation vectors of the markers.
        """
        marker_points = np.array(
            [
                [-self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, -self.marker_size / 2, 0],
                [-self.marker_size / 2, -self.marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        rvecs = []
        tvecs = []
        for corner in corners:
            _, R, t = cv2.solvePnP(
                marker_points,
                corner,
                self.camera.camera_matrix,
                self.camera.dist_coeffs,
                False,
                cv2.SOLVEPNP_ITERATIVE,
            )
            rvecs.append(R)
            tvecs.append(t)
        return np.array(rvecs), np.array(tvecs)

    def get_markers_dict(self) -> None:
        """Get the markers dictionary.

        This function detects the markers in the current frame and estimates their poses.
        It stores the poses in a dictionary with the marker IDs as keys.
        """
        marker_corners, marker_ids = self.detect_markers()
        markers_dict = {}

        if marker_ids is not None:
            rvecs, tvecs = self.estimate_PoseSingleMarkers(marker_corners)

            for i, marker_id in enumerate(marker_ids):
                markers_dict[marker_id[0]] = {
                    "corners": marker_corners[i],
                    "rvec": rvecs[i],
                    "tvec": tvecs[i],
                }

        self.markers_dict = markers_dict

    def stop(self) -> None:
        """Stop the camera and close all windows."""
        self.camera.cap.release()
        cv2.destroyAllWindows()
        self.camera.frame_getter.join()


if __name__ == "__main__":
    camera = Camera()
    aruco_cube_left = ArucoCube(arm="l_arm", camera=camera, marker_size=0.06)
    aruco_cube_right = ArucoCube(arm="r_arm", camera=camera, marker_size=0.03)

    while True:
        left_pose = aruco_cube_left.update_tracker_pose()
        right_pose = aruco_cube_right.update_tracker_pose()
        frame = camera.get_frame_with_cube_pose([left_pose, right_pose])
        cv2.imshow("Cube", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.cap.release()
    cv2.destroyAllWindows()
