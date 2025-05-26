import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore
from trackers.cameras.rgb_camera import RGBCamera  # type: ignore
from trackers.filters.filters import (  # type: ignore
    KalmanFilter3D,
    PoseFilter,
    RotationSmoother,
)
from trackers.tracker import Tracker, TrackerType  # type: ignore
from utils import load_config  # type: ignore


class ArucoTracker(Tracker):
    """Class to detect and track an ArUco cube."""

    def __init__(self, arm: str, camera: RGBCamera) -> None:
        """Initialize the ArUco cube Tracker.

        Args:
            arm (str): Corresponding arm for teleoperation. Either "l_arm" or "r_arm".
            camera (Camera): Camera object to get the frame.
        """
        super().__init__()

        self.tracker_type = TrackerType.ARUCO
        self.arm = arm
        self.camera = camera
        self.frame: Optional[np.ndarray] = None

        self.aruco = self.camera.cv2.aruco

        # get the marker parameters from the config file
        config = load_config("config.yaml")
        self.marker_size = config.get("marker_size", {}).get(self.arm, 0.05)
        self.marker_ids = config.get("marker_ids", {}).get(self.arm, [0, 1, 2, 3, 4, 5])

        # instanciate the aruco detector
        self.aruco_dict = self.aruco.getPredefinedDictionary(self.aruco.DICT_6X6_1000)
        aruco_param = self.aruco.DetectorParameters()
        self.detector = self.aruco.ArucoDetector(self.aruco_dict, aruco_param)

        # add a filter
        self.kf = KalmanFilter3D()
        self.rf = RotationSmoother()
        # self.filter = PoseFilter(0.1)

        # use a filter to smooth the pose
        self.pose_filter = PoseFilter(alpha=0.5)

        self._define_cube(self.marker_ids)

    def _define_cube(self, marker_ids) -> None:
        """Define the cube with specific markers, depending on the arm."""
        c_pt = self.marker_size / 2
        (
            back_marker,
            up_marker,
            front_marker,
            left_marker,
            right_marker,
            down_marker,
        ) = marker_ids

        self.cube_ids = np.array(
            [
                back_marker,
                up_marker,
                front_marker,
                left_marker,
                right_marker,
                down_marker,
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
            ),  # back face
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
                    [-c_pt, -c_pt, -c_pt],
                    [c_pt, -c_pt, -c_pt],
                    [c_pt, c_pt, -c_pt],
                    [-c_pt, c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),  # front face
            np.array(
                [
                    [c_pt, c_pt, c_pt],
                    [c_pt, c_pt, -c_pt],
                    [c_pt, -c_pt, -c_pt],
                    [c_pt, -c_pt, c_pt],
                ],
                dtype=np.float32,
            ),  # left face
            np.array(
                [
                    [-c_pt, c_pt, -c_pt],
                    [-c_pt, c_pt, c_pt],
                    [-c_pt, -c_pt, c_pt],
                    [-c_pt, -c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),  # right face
            np.array(
                [
                    [-c_pt, c_pt, -c_pt],
                    [c_pt, c_pt, -c_pt],
                    [c_pt, c_pt, c_pt],
                    [-c_pt, c_pt, c_pt],
                ],
                dtype=np.float32,
            ),  # down face
        ]

        self.cube = self.aruco.Board(self.cube_corners, self.aruco_dict, self.cube_ids)

    def update_tracker_pose(self) -> Optional[np.ndarray]:
        """Update the cube pose using the camera frame.

        This function detects the markers in the frame and estimates the pose of the cube.
        It applies a filter to the pose and returns the filtered pose.

        Returns:
            tracker_pose (np.ndarray): The pose of the cube in the camera frame.
        """
        try:
            self.frame = self.camera.color_frame[0]
        except IndexError:
            time.sleep(0.01)
            print("No frame")
            return None

        markers_corners, markers_ids = self._detect_markers()

        if markers_ids is not None and len(markers_ids) > 0:
            markers_ids = np.array(markers_ids, dtype=np.int32)

            _, rvec, tvec = self.aruco.estimatePoseBoard(
                markers_corners,
                markers_ids,
                self.cube,
                self.camera.camera_matrix,
                self.camera.dist_coeffs,
                None,
                None,
            )

            # if rvec is not None and tvec is not None:
            #     tracker_pose = np.eye(4)
            #     tracker_pose[:3, :3] = R.from_rotvec(rvec.reshape(1, 3)).as_matrix()

            #     tracker_pose[:3, 3] = tvec.flatten()
            #     tracker_pose_filtered = self.filter.update(tracker_pose)
            #     self.tracker_pose = tracker_pose_filtered
            if rvec is not None and tvec is not None:
                tracker_pose = np.eye(4)
                rotation = R.from_rotvec(rvec.reshape(1, 3)).as_matrix()
                rotation_filtered = self.rf.update(rotation)
                tracker_pose[:3, :3] = rotation_filtered

                tvec_filtered = self.kf.update(tvec.flatten())

                tracker_pose[:3, 3] = tvec_filtered

                tracker_pose[:3, 3] = tvec.flatten()
                filtered_pose = self.pose_filter.update(tracker_pose)
                self.tracker_pose = filtered_pose

        return self.tracker_pose

    def _detect_markers(self) -> tuple[np.ndarray, np.ndarray]:
        """Detect markers in the current frame.

        Returns:
            marker_corners (np.ndarray): The corners of the detected markers.
            marker_ids (np.ndarray): The IDs of the detected markers.
        """
        marker_corners, marker_ids, _ = self.detector.detectMarkers(self.frame)
        return marker_corners, marker_ids

    def _estimate_PoseSingleMarkers(self, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            _, R, t = self.camera.cv2.solvePnP(
                marker_points,
                corner,
                self.camera.camera_matrix,
                self.camera.dist_coeffs,
                False,
                self.camera.cv2.SOLVEPNP_ITERATIVE,
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
            rvecs, tvecs = self._estimate_PoseSingleMarkers(marker_corners)

            for i, marker_id in enumerate(marker_ids):
                markers_dict[marker_id[0]] = {
                    "corners": marker_corners[i],
                    "rvec": rvecs[i],
                    "tvec": tvecs[i],
                }

        self.markers_dict = markers_dict

    def stop(self) -> None:
        """Stop the camera"""
        self.camera.stop()


if __name__ == "__main__":
    import cv2  # type: ignore

    camera = RGBCamera()
    aruco_cube_left = ArucoTracker(arm="l_arm", camera=camera)
    aruco_cube_right = ArucoTracker(arm="r_arm", camera=camera)

    while True:
        left_pose = aruco_cube_left.update_tracker_pose()
        right_pose = aruco_cube_right.update_tracker_pose()
        frame = camera.get_frame_with_cube_pose([left_pose, right_pose])
        cv2.imshow("Cube", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.cap.release()
    cv2.destroyAllWindows()
