import math
import sys
import time

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

import survive
from so100_teleoperation.utils import (
    create_plot,
    fk,
    make_homogenous_matrix_from_rotation_matrix,
    plot_orientation,
    update_plot,
)

# Dictionary mapping your logical tracker names to their serial strings.
trackers = {
    "left_tracker": "LHR-0D914CCE",
    "right_tracker": "LHR-D520271F",
}


def pose_to_matrix(pos, quat):
    """
    Convert a pose given as (position, quaternion) to a 4x4 homogeneous transformation matrix.
    Assumes quaternion is in (w, x, y, z) format.
    scipy's Rotation.from_quat expects [x, y, z, w], so we rearrange.
    """
    # Rearrange quaternion to (x, y, z, w)
    q = [quat[1], quat[2], quat[3], quat[0]]
    rot = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


class ViveTracker:
    def __init__(self, tracker_id):
        self.tracker_name = tracker_id

        if self.tracker_name not in trackers:
            print(
                f"Tracker name '{self.tracker_name}' not found in trackers dictionary. Available trackers: {list(trackers.keys())}"
            )
            sys.exit(1)

        # Initialize our libsurvive-based context (bypassing SteamVR entirely)
        self.vive = survive.triad_survive()

        serial_number = trackers[self.tracker_name]
        print(f"Looking for tracker: {self.tracker_name} (Serial: {serial_number})")

        # Find the matching tracker device by checking if the device name contains the serial number.
        matched_tracker = None
        for dev_name, device in self.vive.devices.items():
            dev_id = (
                device.obj.Name()
            )  # use the underlying object's Name() as an identifier
            print(f"Found device: {dev_name} (ID: {dev_id})")
            if serial_number in dev_id:
                matched_tracker = dev_name
                break

        if matched_tracker is None:
            print(
                f"Tracker with serial '{serial_number}' not found in libsurvive devices."
            )
            sys.exit(1)

        self.tracker_name = matched_tracker
        print(f"Tracker '{tracker_id}' found as '{self.tracker_name}' in libsurvive.")

        self.tracker = self.vive.devices[self.tracker_name]
        self.tracker_pose = np.eye(4)  # current pose (4x4 matrix)
        self.zero_pose = None  # initial reference frame

    def update_tracker_pose(self):
        """Update the tracker pose using pysurvive data."""
        pos, quat = self.tracker.get_pose()  # get_pose() returns (position, quaternion)
        self.tracker_pose = pose_to_matrix(pos, quat)

    def calibrate_pose_zero(self):
        """Sets the current pose as the zero reference frame."""
        print("Calibration: Setting the initial pose as the new coordinate system...")
        time.sleep(2)
        self.update_tracker_pose()
        self.zero_pose = self.tracker_pose  # store initial pose as reference

    def get_relative_tracker_pose(self):
        """Compute the tracker pose relative to the calibrated (zero) frame."""
        self.update_tracker_pose()

        if self.zero_pose is None:
            print("Warning: Zero pose not set. Returning absolute pose.")
            return self.tracker_pose  # Return absolute pose if not calibrated

        relative_pose = np.linalg.inv(self.zero_pose) @ self.tracker_pose

        # Apply additional fixed rotations as in your original code.
        rotation = R.from_euler("xyz", [0, 0, -30], degrees=True).as_matrix()
        Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
        relative_pose = Trot @ relative_pose

        rotation = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
        Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
        relative_pose = Trot @ relative_pose

        return relative_pose

    def get_tracker_position(self) -> npt.ArrayLike:
        """Return the tracker’s current position relative to the calibrated frame."""
        relative_pose = self.get_relative_tracker_pose()
        return relative_pose[:3, 3]

    def get_tracker_euler_angles(self) -> npt.ArrayLike:
        """Return the tracker’s orientation (roll, pitch, yaw) in degrees relative to the calibrated frame."""
        relative_pose = self.get_relative_tracker_pose()
        return R.from_matrix(relative_pose[:3, :3]).as_euler("xyz", degrees=True)


if __name__ == "__main__":
    # For example, create a ViveTracker for "left_tracker"

    # Initialize our libsurvive context
    survive_manager = survive.triad_survive()
    print("Detected devices:")
    print(survive_manager.devices)
    for device_name, device in survive_manager.devices.items():
        pos, quat = device.get_pose()  # (position, quaternion)
        print(f"Device '{device_name}': Position = {pos}, Quaternion = {quat}")

    exit(1)
    tracker = ViveTracker("left_tracker")

    # Optionally calibrate the coordinate system:
    # tracker.calibrate_pose_zero()

    while True:
        tracker.update_tracker_pose()
        pose = tracker.tracker_pose
        print("Absolute Tracker Pose (4x4 matrix):")
        print(pose)

        position = tracker.get_tracker_position()
        orientation = tracker.get_tracker_euler_angles()
        print(f"Relative Position: {position}")
        print(f"Relative Orientation (Euler XYZ, degrees): {orientation}")

        time.sleep(0.1)
