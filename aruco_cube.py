import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import threading
from collections import deque

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
                self.frame.append(frame)
            time.sleep(0.005)


class ArucoCube:
    def __init__(self, marker_size=0.04):

        self.camera = Camera()
        self.frame = None

        self.marker_size = marker_size

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        aruco_param = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco_param)

        self.cube = None
        self.define_cube()
        self.cube_pose = None

        self.markers_dict = {}

    def define_cube(self):
        c_pt = self.marker_size/2
        self.cube_ids = np.arange(6)
        self.cube_corners = [
            np.array([[-c_pt, c_pt, c_pt], [c_pt, c_pt, c_pt], [c_pt, -c_pt, c_pt], [-c_pt, -c_pt, c_pt]], dtype=np.float32),  # ID 0
            np.array([[-c_pt, -c_pt, c_pt], [c_pt, -c_pt, c_pt], [c_pt, -c_pt, -c_pt], [-c_pt, -c_pt, -c_pt]], dtype=np.float32),  # ID 1
            np.array([[-c_pt, c_pt, -c_pt], [-c_pt, c_pt, c_pt], [-c_pt, -c_pt, c_pt], [-c_pt, -c_pt, -c_pt]], dtype=np.float32),  # ID 2
            np.array([[-c_pt, c_pt, -c_pt], [c_pt, c_pt, -c_pt], [c_pt, c_pt, c_pt], [-c_pt, c_pt, c_pt]], dtype=np.float32),  # ID 3
            np.array([[c_pt, c_pt, c_pt], [c_pt, c_pt, -c_pt], [c_pt, -c_pt, -c_pt], [c_pt, -c_pt, c_pt]], dtype=np.float32),  # ID 4
            np.array([[-c_pt, -c_pt, -c_pt], [c_pt, -c_pt, -c_pt], [c_pt, c_pt, -c_pt], [-c_pt, c_pt, -c_pt]], dtype=np.float32),   # ID 5
        ]
        self.cube = aruco.Board(self.cube_corners, self.aruco_dict, self.cube_ids)

    def update_cube_pose(self):
        try:
            self.frame = self.camera.frame[0]
        except IndexError:
            time.sleep(0.01)
            print("No frame")
            return False

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
                None
            )

            if rvec is not None and tvec is not None:
                cube_pose = np.eye(4)
                cube_pose[:3,:3] = R.from_rotvec(rvec.reshape(1,3)).as_matrix()

                cube_pose[:3,3] = tvec.flatten()
                self.cube_pose = cube_pose
            return True

    def detect_markers(self):
        marker_corners, marker_ids, _ = self.detector.detectMarkers(self.frame)
        return marker_corners, marker_ids

    def estimate_PoseSingleMarkers(self, corners):
        marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                    [self.marker_size / 2, self.marker_size / 2, 0],
                                    [self.marker_size / 2, -self.marker_size / 2, 0],
                                    [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
        rvecs = []
        tvecs = []
        for corner in corners:
            _, R, t = cv2.solvePnP(marker_points, corner, self.camera.camera_matrix, self.camera.dist_coeffs, False, cv2.SOLVEPNP_ITERATIVE)
            rvecs.append(R)
            tvecs.append(t)
        return np.array(rvecs), np.array(tvecs)

    def get_markers_dict(self):
        marker_corners, marker_ids = self.detect_markers(self.frame)
        markers_dict = {}

        if marker_ids is not None:
            rvecs, tvecs = self.estimate_PoseSingleMarkers(marker_corners, self.camera.camera_matrix, self.camera.dist_coeffs)

            for i, marker_id in enumerate(marker_ids):
                markers_dict[marker_id[0]] = {
                    'corners': marker_corners[i],
                    'rvec': rvecs[i],
                    'tvec': tvecs[i]
                }

        self.markers_dict = markers_dict

    def show_cube_infos(self):
        if self.cube_pose is not None:
            rvec = R.from_matrix(self.cube_pose[:3,:3]).as_rotvec()
            tvec = self.cube_pose[:3,3]
            cv2.drawFrameAxes(self.frame, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.03)
            
            cv2.putText(
                self.frame,
                f"Cube - x: {np.round(tvec[0],3)}, y: {np.round(tvec[1],3)}, z: {np.round(tvec[2],3)}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            roll, pitch, yaw = R.from_matrix(self.cube_pose[:3,:3]).as_euler('xyz', degrees=True)

            cv2.putText(
                self.frame,
                f"Cube - roll: {np.round(roll)}, pitch: {np.round(pitch)}, yaw: {np.round(yaw)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
        return self.frame

    def show_markers_infos(self, frame, markers_dict):
        markers_corners, markers_ids = self.detect_markers(frame)
        aruco.drawDetectedMarkers(frame, markers_corners, markers_ids)
        for marker_id, data in markers_dict.items():
            corners = data['corners']
            rvec = data['rvec']
            tvec = data['tvec']

            cv2.drawFrameAxes(frame, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.03)
            translation = tvec.flatten()
            cv2.putText(
                frame,
                f"{marker_id} - x: {np.round(translation[0],3)}, y: {np.round(translation[1],3)}, z: {np.round(translation[2],3)}",
                (10, 20 + 20 * marker_id),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
        # cv2.aruco.drawDetectedMarkers(frame, [corners], np.array([[marker_id]]))

        return frame

    def stop(self):
        self.camera.cap.release()
        cv2.destroyAllWindows()
        self.camera.frame_getter.join()

if __name__ == "__main__":
    aruco_cube = ArucoCube()

    while True:
        success = aruco_cube.update_cube_pose()
        if success:
            print("success")
            aruco_cube.show_cube_infos()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cap.release()
    cv2.destroyAllWindows()
