import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R

class Aruco:
    def __init__(self, marker_size=0.04):
        self.marker_size = marker_size
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        aruco_param = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco_param)

        self.markers_dict = {}
        self.cube = None
        self.define_cube()

        self.cube_pose = None

    def define_cube(self):
        c_pt = self.marker_size/2
        self.cube_ids = np.arange(6)
        self.cube_corners = [
            np.array([[-c_pt, c_pt, c_pt], [c_pt, c_pt, c_pt], [c_pt, -c_pt, c_pt], [-c_pt, -c_pt, c_pt]], dtype=np.float32),  # ID 0
            np.array([[-c_pt, -c_pt, c_pt], [c_pt, -c_pt, c_pt], [c_pt, -c_pt, -c_pt], [-c_pt, -c_pt, -c_pt]], dtype=np.float32),  # ID 1
            np.array([[-c_pt, c_pt, -c_pt], [-c_pt, c_pt, c_pt], [-c_pt, -c_pt, c_pt], [-c_pt, -c_pt, -c_pt]], dtype=np.float32),  # ID 2
            np.array([[-c_pt, c_pt, -c_pt], [c_pt, c_pt, -c_pt], [c_pt, c_pt, c_pt], [-c_pt, c_pt, c_pt]], dtype=np.float32),  # ID 3
            np.array([[c_pt, c_pt, c_pt], [c_pt, c_pt, -c_pt], [c_pt, -c_pt, -c_pt], [c_pt, -c_pt, c_pt]], dtype=np.float32),  # ID 4
            np.array([[-c_pt, -c_pt, -c_pt], [c_pt, -c_pt, -c_pt], [c_pt, c_pt, -c_pt], [-c_pt, c_pt, -c_pt]], dtype=np.float32)   # ID 5
        ]
        self.cube = aruco.Board(self.cube_corners, self.aruco_dict, self.cube_ids)

    def detect_markers(self, image):
        marker_corners, marker_ids, _ = self.detector.detectMarkers(image)
        return marker_corners, marker_ids

    def estimate_PoseSingleMarkers(self, corners, camera_matrix, dist_coeffs):
        marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                    [self.marker_size / 2, self.marker_size / 2, 0],
                                    [self.marker_size / 2, -self.marker_size / 2, 0],
                                    [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
        rvecs = []
        tvecs = []
        for c in corners:
            _, R, t = cv2.solvePnP(marker_points, c, camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_ITERATIVE)
            rvecs.append(R)
            tvecs.append(t)
        return np.array(rvecs), np.array(tvecs)

    def get_markers_dict(self, image, camera_matrix, dist_coeffs):
        marker_corners, marker_ids = self.detect_markers(image)
        markers_dict = {}

        if marker_ids is not None:
            rvecs, tvecs = self.estimate_PoseSingleMarkers(marker_corners, camera_matrix, dist_coeffs)

            for i, marker_id in enumerate(marker_ids):
                markers_dict[marker_id[0]] = {
                    'corners': marker_corners[i],
                    'rvec': rvecs[i],
                    'tvec': tvecs[i]
                }

        self.markers_dict = markers_dict

    def get_cube_pose(self, frame, camera_matrix, dist_coeffs):
        markers_corners, markers_ids = self.detect_markers(frame)
        if markers_ids is not None and len(markers_ids) > 0:
            markers_ids = np.array(markers_ids, dtype=np.int32)

            _, rvec, tvec = cv2.aruco.estimatePoseBoard(
                markers_corners,
                markers_ids,
                self.cube,
                camera_matrix,
                dist_coeffs,
                None,
                None
            )

            if rvec is not None and tvec is not None:
                cube_pose = np.eye(4)
                cube_pose[:3,:3], _ = cv2.Rodrigues(rvec)
                cube_pose[:3,3] = tvec.flatten()
                self.cube_pose = cube_pose
                return True

        self.cube_pose = None
        return False

    def show_cube_infos(self, frame, camera_matrix, dist_coeffs):
        if self.cube_pose is not None:
            rvec = R.from_matrix(self.cube_pose[:3,:3]).as_rotvec()
            tvec = self.cube_pose[:3,3]
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            cv2.putText(
                frame,
                f"Cube - x: {np.round(tvec[0],3)}, y: {np.round(tvec[1],3)}, z: {np.round(tvec[2],3)}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
        return frame


    def show_markers_infos(self, frame, markers_dict, camera_matrix, dist_coeffs):
        markers_corners, markers_ids = self.detect_markers(frame)
        aruco.drawDetectedMarkers(frame, markers_corners, markers_ids)
        for marker_id, data in markers_dict.items():
            corners = data['corners']
            rvec = data['rvec']
            tvec = data['tvec']

            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
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


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros((5, 1))
        self.calibrate_camera()

    def calibrate_camera(self):
        frame = self.get_frame()
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        else:
            return frame


if __name__ == "__main__":
    aruco_cube = Aruco()
    camera = Camera()

    while True:
        frame = camera.get_frame()
        success = aruco_cube.get_cube_pose(frame, camera.camera_matrix, camera.dist_coeffs)
        if success:
            frame = aruco_cube.show_cube_infos(frame, camera.camera_matrix, camera.dist_coeffs)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cap.release()
    cv2.destroyAllWindows()
