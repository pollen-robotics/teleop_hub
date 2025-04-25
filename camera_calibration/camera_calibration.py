import argparse

# import json
import os
import time
from typing import Optional

import cv2  # type: ignore
import numpy as np
from config_utils import update_config

ARUCO_DICT = cv2.aruco.DICT_4X4_1000  # Dictionary ID
SQUARES_X = 11  # Number of squares horizontally
SQUARES_Y = 8  # Number of squares vertically
SQUARE_LENGTH = 20.75  # Square side length (in mm)
MARKER_LENGTH = 15.58  # ArUco marker side length (in mm)
LEGACY_PATTERN = True  # True if the board starts with a black box in the upper left corner

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_tracker", "config.yaml"))


def get_camera_capture(usb_cam: bool = True, cam_id: str = "0") -> cv2.VideoCapture:
    """
    Get the camera capture based on the camera type (USB or IP).

    Args:
        - usb_cam: True if using a USB camera, False if using an IP camera
        - cam_id: Camera ID or IP address

    Returns:
        - cap: The camera capture object
    """
    if usb_cam:
        try:
            cam_index = int(cam_id)
            print("Using USB camera with ID:", cam_index)
            cap = cv2.VideoCapture(cam_index)
        except Exception:
            print("Error: Unable to open USB camera. Camera ID needs to be the device index.")
            return None
    else:
        try:
            cam_ip = f"http://{cam_id}:8080/video"
            cap = cv2.VideoCapture(cam_ip)
        except Exception:
            print("Error: Unable to open IP camera. Please check the IP address of your camera.")
            return None
    return cap


def define_aruco_tools(
    aruco_dict: cv2.aruco.Dictionary = ARUCO_DICT,
    squares_x: int = SQUARES_X,
    squares_y: int = SQUARES_Y,
    square_length: float = SQUARE_LENGTH,
    marker_length: float = MARKER_LENGTH,
    legacy_pattern: bool = LEGACY_PATTERN,
) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.ArucoDetector]:
    """
    Define the ArUco board and detector with the given parameters.

    Args:
        - aruco_dict: ArUco Dictionary ID
        - squares_x: Number of squares horizontally
        - squares_y: Number of squares vertically
        - square_length: Square side length (in mm)
        - marker_length: ArUco marker side length (in mm)
        - legacy_pattern: True if the board starts with a black box in the upper left corner

    Returns:
        - board: The defined ArUco board
        - detector: The ArUco detector
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    board.setLegacyPattern(legacy_pattern)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return board, detector


def extract_marker_info(
    image: np.ndarray, detector: cv2.aruco.ArucoDetector, board: cv2.aruco.CharucoBoard
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract marker information from the image using the ArUco detector.

    Args:
        - image: The input image in grayscale
        - detector: The ArUco detector

    Returns:
        - marker_corners: The detected marker corners
        - marker_ids: The detected marker IDs
    """
    marker_corners, marker_ids, marker_centers = None, None, None
    marker_corners, marker_ids, rejected_corners = detector.detectMarkers(image)
    if marker_ids is not None and len(marker_ids) > 5:
        marker_corners, marker_ids, _, _ = detector.refineDetectedMarkers(
            image, board, marker_corners, marker_ids, rejected_corners
        )

        marker_centers = np.mean([np.mean(corner, axis=1) for corner in marker_corners], axis=0)

    return marker_corners, marker_ids, marker_centers


def extract_charuco_info(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    marker_corners: np.ndarray,
    marker_ids: np.ndarray,
) -> tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract Charuco corners from the image using the ArUco detector.

    Args:
        - image: The input image
        - board: The defined ArUco board
        - marker_corners: The detected marker corners
        - marker_ids: The detected marker IDs

    Returns:
        - ret: The return value of the interpolation
        - charucoCorners: The detected Charuco corners
        - charucoIds: The detected Charuco IDs
    """
    print("extract_charuco_info")
    return cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)


def is_far_enough(
    marker_centers: np.ndarray,
    last_marker_center: Optional[np.ndarray],
    movement_threshold: int,
) -> bool:
    """
    Check if the detected marker centers are far enough from the last detected center.

    Args:
        - marker_centers: The detected marker centers
        - last_marker_center: The last detected marker center
        - movement_threshold: The threshold for significant movement

    Returns:
        - bool: True if the markers have moved significantly, False otherwise
    """
    if last_marker_center is None:
        return True
    return bool(np.linalg.norm(marker_centers - last_marker_center) > movement_threshold)


def get_calibration_parameters(
    cap,
    images_nb: int = 20,
    board: cv2.aruco.CharucoBoard = None,
    detector: cv2.aruco.ArucoDetector = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get camera matrix and distorsion coefficients using a Charuco board.
    Uses several images with the board in different positions to calibrate the camera.
    The board needs to be in different positions, angles and distances to the camera.

    Args:
        - usb_cam : True if using a USB camera, False if using an IP camera (such as a smartphone)
            Defaults to True.
        - cam_id:
            - device index for USB camera ('0' for the integrated one, '-1' for the last plugged one)
            - IP address of the camera for IP camera
        - images_nb: Number of images to capture for calibration. Defaults to 20.

    Returns:
        - camera_matrix: Camera intrinsic matrix (3x3)
        - dist_coeffs: Distortion coefficients (list of 5 values)
    """
    image_id = 0
    all_charuco_ids, all_charuco_corners = [], []

    # Define a threshold to capture images only if the board moves significantly
    last_marker_center = None
    movement_threshold = 100

    # Loop over images and extraction of corners
    while image_id < images_nb:
        success, image = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, marker_centers = extract_marker_info(gray, detector, board)

        if marker_centers is not None and is_far_enough(marker_centers, last_marker_center, movement_threshold):
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

            if ret > 0 and charucoIds is not None and len(charucoIds) >= 6:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
                last_marker_center = marker_centers
                image_id += 1
                print("Captured image", image_id, "of", images_nb)

        cv2.imshow("image", image)
        cv2.waitKey(1)

    all_charuco_corners = [np.array(corner, dtype=np.float32) for corner in all_charuco_corners]
    all_charuco_ids = [np.array(ids, dtype=np.int32) for ids in all_charuco_ids]

    # Calibrate camera with extracted information
    _, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, gray.shape, None, None
    )

    return camera_matrix, dist_coeffs


# def save_calibration_parameters(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, filename: str) -> None:
#     """
#     Save camera matrix and distortion coefficients in a JSON file. It will be saved in the
#     camera_parameters folder with the name of the camera.

#     Args:
#         - camera_matrix: Camera intrinsic matrix (3x3)
#         - dist_coeffs: Distortion coefficients (list of 5 values)
#         - filename: Name of the JSON file to save the parameters
#     """
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     camera_parameters_dir = os.path.join(current_dir, "camera_parameters")
#     filepath = os.path.join(camera_parameters_dir, f"camera_{filename}.json")

#     camera_params = {
#         "camera_matrix": camera_matrix.tolist(),
#         "dist_coeffs": dist_coeffs.tolist(),
#     }
#     # if the file already exists, replace the old values

#     if not os.path.exists(camera_parameters_dir):
#         os.makedirs(camera_parameters_dir)
#     # if the file already exists, replace the old values
#     # if the file does not exist, create it
#     mode = "w" if os.path.exists(filepath) else "w+"
#     with open(filepath, mode) as f:
#         json.dump(camera_params, f)
#     print("Calibration parameters saved to", filepath)


def main(
    usb_cam: bool = True,
    cam_id: str = "0",
    images_nb: int = 20,
    aruco_dict: cv2.aruco.Dictionary = ARUCO_DICT,
    squares_x: int = SQUARES_X,
    squares_y: int = SQUARES_Y,
    square_length: float = SQUARE_LENGTH,
    marker_length: float = MARKER_LENGTH,
    legacy_pattern: bool = LEGACY_PATTERN,
) -> None:
    """
    Main function to calibrate the camera using a Charuco board and get the camera matrix and
    distortion coefficients. It will save the parameters in a JSON file in the camera_parameters
    folder with the name of the camera.

    Args:
        - usb_cam: True if using a USB camera, False if using an IP camera (such as a smartphone)
        - cam_id:
            - device index for USB camera ('0' for the integrated one, '-1' for the last plugged one)
            - IP address of the camera for IP camera
        - images_nb: Number of images to capture for calibration. Defaults to 20.
        - aruco_dict: ArUco Dictionary ID
        - squares_x: Number of squares horizontally on the board
        - squares_y: Number of squares vertically on the board
        - square_length: Square side length (in mm)
        - marker_length: ArUco marker side length (in mm)
        - legacy_pattern: True if the board starts with a black box in the upper left corner
    """
    board, detector = define_aruco_tools(
        aruco_dict, squares_x, squares_y, square_length, marker_length, legacy_pattern
    )
    print("ArUco board and detector defined.")
    cap = get_camera_capture(usb_cam, cam_id)
    while cap is None:
        time.sleep(0.5)
        cap = get_camera_capture(usb_cam, cam_id)
    time.sleep(3)
    print("Camera capture initialized.")

    camera_matrix, dist_coeffs = get_calibration_parameters(cap, images_nb, board, detector)
    # save_calibration_parameters(camera_matrix, dist_coeffs, cam_id)
    update_config(
        CONFIG_PATH,
        {
            "camera": {
                "with_calibration": True,
                "camera_id": cam_id,
                "usb_mode": usb_cam,
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.tolist(),
            }
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration using Charuco board")
    parser.add_argument(
        "--usb_cam",
        type=bool,
        default=True,
        help="USB camera (True) or IP camera (False) mode",
    )
    parser.add_argument("--cam_id", type=str, default="0", help="Camera index or IP address")
    parser.add_argument("--images_nb", type=int, default=20, help="Number of images for calibration")
    parser.add_argument(
        "--aruco_dict",
        type=int,
        default=ARUCO_DICT,
        help="ArUco Dictionary ID",
    )
    parser.add_argument(
        "--squares_x",
        type=int,
        default=SQUARES_X,
        help="Number of squares horizontally",
    )
    parser.add_argument(
        "--squares_y",
        type=int,
        default=SQUARES_Y,
        help="Number of squares vertically",
    )
    parser.add_argument(
        "--square_length",
        type=float,
        default=SQUARE_LENGTH,
        help="Square side length (in mm)",
    )
    parser.add_argument(
        "--marker_length",
        type=float,
        default=MARKER_LENGTH,
        help="ArUco marker side length (in mm)",
    )
    parser.add_argument(
        "--legacy_pattern",
        type=bool,
        default=LEGACY_PATTERN,
        help="Legacy pattern for board",
    )
    args = parser.parse_args()

    main(
        usb_cam=args.usb_cam,
        cam_id=args.cam_id,
        images_nb=args.images_nb,
        aruco_dict=args.aruco_dict,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        legacy_pattern=args.legacy_pattern,
    )

    cv2.destroyAllWindows()
