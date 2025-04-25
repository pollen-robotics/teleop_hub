import sys

import numpy as np
import trackers.vive_tracker.triad_openvr as triad_openvr  # type: ignore
from trackers.tracker import Tracker  # type: ignore

trackers = {
    "l_arm": "LHR-0D914CCE",
    "r_arm": "LHR-D520271F",
}


class ViveTracker(Tracker):
    def __init__(self, tracker_id):
        super().__init__(tracker_id)
        self.tracker_name = tracker_id

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

    def convert_openvr_matrix(self, hmd_matrix):
        """
        Convertit une matrice OpenVR 3x4 (HmdMatrix34_t) en une matrice 4x4 NumPy.

        :param hmd_matrix: Matrice OpenVR (hmd_matrix.mDeviceToAbsoluteTracking)
        :return: Matrice 4x4 NumPy
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

    def update_tracker_pose(self):
        pose = self.tracker.get_pose_matrix()
        self.tracker_pose = self.convert_openvr_matrix(pose)

    def stop(self):
        print("Tracking stopped.")
