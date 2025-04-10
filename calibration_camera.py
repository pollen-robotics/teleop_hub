import json
import time

import cv2  # type: ignore
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_1000    # Dictionary ID
SQUARES_X = 11                          # Number of squares horizontally
SQUARES_Y = 8                           # Number of squares vertically
SQUARE_LENGTH = 20.75                   # Square side length (in mm)
MARKER_LENGTH = 15.58                   # ArUco marker side length (in mm)
LEGACY_PATTERN = True                   # True if the board starts with a black box in the upper left


def get_calibration_parameters(cam_ip: str, images_nb: int = 20):
    if cam_ip == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        complete_id = f"http://{cam_ip}:8080/video"
        print(f"Connecting to camera at {complete_id}")
        cap = cv2.VideoCapture(complete_id)
    time.sleep(3)

    # Define the aruco dictionary, charuco board and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    board.setLegacyPattern(LEGACY_PATTERN)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    image_id = 0

    all_charuco_ids = []
    all_charuco_corners = []

    # Define a threshold to capture images with the board in different places
    last_marker_center = None
    movement_threshold = 100.0

    # Loop over images and extraction of corners
    while image_id < images_nb:
        success, image = cap.read()
        if not success:
            continue

        # Detect markers on a grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected_corners = detector.detectMarkers(gray)

        if marker_ids is not None and len(marker_ids) > 5:
            print(f"Image {image_id + 1}: Found {len(marker_ids)} markers")

            marker_corners, marker_ids, _, _ = detector.refineDetectedMarkers(
                gray, board, marker_corners, marker_ids, rejected_corners
            )

            marker_centers = np.mean([np.mean(corner, axis=1) for corner in marker_corners], axis=0)

            if last_marker_center is None or np.linalg.norm(marker_centers - last_marker_center) > movement_threshold:
                ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board
                )
                cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

                if ret > 0 and charucoIds is not None and len(charucoIds) >= 6:
                    all_charuco_corners.append(charucoCorners)
                    all_charuco_ids.append(charucoIds)
                    last_marker_center = marker_centers
                    image_id += 1

        cv2.imshow('image', image)
        cv2.waitKey(1)

    all_charuco_corners = [np.array(corner, dtype=np.float32) for corner in all_charuco_corners]
    all_charuco_ids = [np.array(ids, dtype=np.int32) for ids in all_charuco_ids]

    # Calibrate camera with extracted information
    _, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, gray.shape, None, None
    )

    return mtx, dist


def save_calibration_parameters(mtx, dist, filename: str):
    # Save camera matrix and distortion coefficients in JSON file
    filepath = f'./cameras_param/{filename}.json'
    camera_params = {
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(camera_params, f)
    print("Calibration parameters saved to", filepath)


if __name__ == "__main__":
    camera_ip = 'webcam'
    # Get calibration parameters
    mtx, dist = get_calibration_parameters(camera_ip)
    # Save calibration parameters to JSON file
    save_calibration_parameters(mtx, dist, camera_ip)
