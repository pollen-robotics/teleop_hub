import cv2
import cv2.aruco as aruco
import numpy as np

class Aruco:
    def __init__(self, marker_size=0.03):
        self.marker_size = marker_size
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        self.aruco_param = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_param)

        self.markers_dict = {}

    def detect_markers(self, image):
        marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(image)
        return marker_corners, marker_ids

    def estimate_PoseSingleMarkers(self, corners, camera_matrix, dist_coeffs):
        marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                    [self.marker_size / 2, self.marker_size / 2, 0],
                                    [self.marker_size / 2, -self.marker_size / 2, 0],
                                    [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
        rvecs = []
        tvecs = []
        i = 0
        for c in corners:
            _, R, t = cv2.solvePnP(marker_points, corners[i], camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_ITERATIVE)
            rvecs.append(R)
            tvecs.append(t)
        return np.array(rvecs), np.array(tvecs)

    def get_markers_infos(self, image, camera_matrix, dist_coeffs):
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

    def show_markers_infos(self, frame, markers_dict, camera_matrix, dist_coeffs):
        for marker_id, data in markers_dict.items():
            corners = data['corners']
            rvec = data['rvec']
            tvec = data['tvec']

            cv2.aruco.drawDetectedMarkers(frame, [corners], np.array([[marker_id]]))
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            translation = tvec.flatten()
            cv2.putText(frame, f"{marker_id} - x: {np.round(translation[0],3)}, y: {np.round(translation[1],3)}, z: {np.round(translation[2],3)}",
                        (10, 20 + 20 * marker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
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
        aruco_cube.get_markers_infos(frame, camera.camera_matrix, camera.dist_coeffs)
        if aruco_cube.markers_dict is not None:
            frame = aruco_cube.show_markers_infos(frame, aruco_cube.markers_dict, camera.camera_matrix, camera.dist_coeffs)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cap.release()
    cv2.destroyAllWindows()
