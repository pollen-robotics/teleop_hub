import sys

import numpy as np
import trackers.vive_tracker.triad_openvr as triad_openvr  # type: ignore
from openvr import HmdMatrix34_t  # type: ignore
from trackers.tracker import Tracker, TrackerType  # type: ignore
from utils import load_config  # type: ignore

# trackers = {
#     "l_arm": "LHR-0D914CCE",
#     "r_arm": "LHR-D520271F",
# }


class ViveTracker(Tracker):
    """Class for tracking Vive trackers."""

    def __init__(self, tracker_id: str) -> None:
        """Initialize the ViveTracker class.

        Args:
            tracker_id (str): The ID of the tracker to be used.
        """
        super().__init__()
        self.tracker_name = tracker_id
        self.tracker_type = TrackerType.VIVE
        config = load_config("config.yaml")
        trackers = config["vive_trackers"]

        if self.tracker_name not in trackers:
            print(
                f"Tracker name '{self.tracker_name}' not found in trackers dictionary. Available trackers:"
                f" {list(trackers.keys())}"
            )
            sys.exit(1)

        self.vive = triad_openvr.triad_openvr()

        serial_number = trackers[self.tracker_name]
        print(f"Looking for tracker: {self.tracker_name} (Serial: {serial_number})")

        # Find corresponding tracker device in SteamVR
        matched_tracker = None
        for dev in self.vive.devices:
            dev_serial = self.vive.devices[dev].get_serial().decode("utf-8").strip()
            print(f"Found device: {dev} (Serial: {dev_serial})")
            if dev_serial == serial_number:
                matched_tracker = dev
                break

        if matched_tracker is None:
            print(f"Tracker with serial '{serial_number}' not found in SteamVR.")
            sys.exit(1)

        self.tracker_name = matched_tracker

        print(f"Tracker '{tracker_id}' found as '{self.tracker_name}' in SteamVR.")

        self.tracker = self.vive.devices[self.tracker_name]
        self.tracker_euler_angles = np.zeros(3)
        self.tracker_position = np.zeros(3)
        self.tracker_pose = np.eye(4)  # Current pose
        self.zero_pose = None  # Initial reference frame

    def update_tracker_pose(self) -> None:
        """Update the tracker pose."""
        pose = self.tracker.get_pose_matrix()
        self.tracker_pose = self._convert_openvr_matrix(pose)

    def _convert_openvr_matrix(self, hmd_matrix: HmdMatrix34_t) -> np.ndarray:
        """Converts an OpenVR 3x4 matrix (HmdMatrix34_t) to a 4x4 NumPy matrix.

        Args:
            hmd_matrix (HmdMatrix34_t): OpenVR 3x4 matrix (HmdMatrix34_t) to be converted.

        Returns:
            np.ndarray: Converted 4x4 NumPy matrix.
        """
        m = np.array(
            [
                [
                    hmd_matrix[0][0],
                    hmd_matrix[0][1],
                    hmd_matrix[0][2],
                    hmd_matrix[0][3],
                ],
                [
                    hmd_matrix[1][0],
                    hmd_matrix[1][1],
                    hmd_matrix[1][2],
                    hmd_matrix[1][3],
                ],
                [
                    hmd_matrix[2][0],
                    hmd_matrix[2][1],
                    hmd_matrix[2][2],
                    hmd_matrix[2][3],
                ],
                [0, 0, 0, 1],
            ]
        )
        return m

    def stop(self):
        print("Tracking stopped.")
