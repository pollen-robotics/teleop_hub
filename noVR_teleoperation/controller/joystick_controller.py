import threading
import time
from typing import Optional

import numpy as np
from camera.camera import Camera  # type: ignore
from controller.arduino import ArduinoController  # type: ignore
from controller.controller import Controller  # type: ignore
from controller.feetech import Feetech  # type: ignore
from filter.filters import PoseFilter  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from trackers.aruco_tracker.aruco_cube import ArucoCube  # type: ignore
from trackers.tracker import TrackerType  # type: ignore
from trackers.vive_tracker.vive_tracker import ViveTracker  # type: ignore
from utils import (  # type: ignore
    load_config,
    make_homogenous_matrix_from_rotation_matrix,
)

feetech_ports = {
    "l_arm": "/dev/noVR_left_motor",
    "r_arm": "/dev/noVR_right_motor",
}

arduino_ports = {
    "l_arm": "/dev/noVR_left_arduino",
    "r_arm": "/dev/noVR_right_arduino",
}

gripper_joints = {"l_arm": [65, 30], "r_arm": [-65, -30]}

FEETECH_GRIPPER = "feetech"
POTENTIOMETER_GRIPPER = "potentiometer"

class JoystickController(Controller):
    """Joystick controller class.

    This class is responsible for controlling a joystick-type controller, thanks to a tracker object (Vive or ArUco),
    a Arduino object for the joystick and the buttons, and a Feetech object for the gripper.

    It is a subclass of the Controller class.
    """

    def __init__(
        self,
        tracker_type: TrackerType,
        arm: str,
        camera: Optional[Camera],
    ) -> None:
        """Initialize the joystick controller.

        Args:
            tracker_type (TrackerType): The type of tracker to use (Vive or ArUco).
            arm (str): The arm to control ("l_arm" or "r_arm").
            camera (Camera, optional): The camera object for ArUco tracking. Defaults to None.
        """
        super().__init__()
        self.tracker_type = tracker_type
        self.arm = arm

        self.gripper_type = load_config("config.yaml")["gripper_type"]
        print(f"Gripper type: {self.gripper_type}")
        
            
        self.arduino = ArduinoController(arduino_ports[arm], self.gripper_type)

        if self.tracker_type == TrackerType.ARUCO:
            self.camera = camera
            if self.camera is None:
                raise ValueError("Camera is required for ArUco tracking.")

            self.tracker = ArucoCube(arm, self.camera)
            self.pose_filter = PoseFilter(alpha=0.5)
            self.is_filtered = True

        elif self.tracker_type == TrackerType.VIVE:
            self.tracker = ViveTracker(arm)
            self.is_filtered = False

        # self.gripper = Feetech(feetech_ports[arm])
        self.joystick_x = None
        self.joystick_y = None
        self.joystick_button = None
        self.buttonA = None
        self.buttonB = None

        self.stop_flag = False
        if self.gripper_type == FEETECH_GRIPPER:
            self.gripper = Feetech(feetech_ports[arm])
            self.gripper_joints_limit = load_config("config.yaml")["feetech_gripper_joints_limit"][
                self.arm
            ]
        elif self.gripper_type == POTENTIOMETER_GRIPPER:
            self.gripper_joints_limit = load_config("config.yaml")["potentiometer_gripper_joints_limit"][
                self.arm
            ]
        else:
            raise ValueError(f"Unknown gripper type: {self.gripper_type}")

        thread = threading.Thread(target=self._update_arduino_data)
        thread.daemon = True
        thread.start()

        # self.init_controller()
        # self.init_gripper(self.gripper_joints_limit[1])

    def init_gripper(self, joint):
        self.gripper.enable_torque()
        self.gripper.goto_joints([joint], 1)
        print([joint])
        print("iciiiiiiiiiiiiiiiiiiii")
        self.gripper.disable_torque()

    def init_controller(self) -> None:
        """Initialize the controller.

        This method initializes the tracker and the gripper.
        """
        while self.tracker.tracker_pose is None:
            self.tracker.update_tracker_pose()
            time.sleep(0.1)
        self.tracker_init_pose = self.tracker.tracker_pose
        if self.gripper_type == FEETECH_GRIPPER:
            self.init_gripper(self.gripper_joints_limit[1])

    def _update_arduino_data(self) -> None:
        """Update the Arduino data in a separate thread.

        This method reads the joystick data from the Arduino and updates the joystick position and button states.
        """
        while not self.stop_flag:
            arduino_values = self.arduino.read()
            # x, y, button_cmd, buttonA, buttonB, potentiometer = self.arduino.read()
            if arduino_values[0] is not None:
                self.joystick_button = arduino_values[2]
                if self.arm == "r_arm":
                    self.buttonA = arduino_values[3]
                    self.buttonB = arduino_values[4]
                    self.joystick_x = arduino_values[0]
                    self.joystick_y = arduino_values[1]
                else:
                    self.buttonA = arduino_values[4]
                    self.buttonB = arduino_values[3]
                    self.joystick_x = 1024 - arduino_values[0]
                    self.joystick_y = 1024 - arduino_values[1]
                if self.gripper_type == POTENTIOMETER_GRIPPER:
                    self.potentiometer = arduino_values[5]
                    print(self.potentiometer)

            else:
                print("No data")
            time.sleep(0.1)

    def get_gripper_joint(self):
        if self.gripper_type == FEETECH_GRIPPER:
            gripper_joint = self.gripper.get_joints()[0]
        if self.gripper_type == POTENTIOMETER_GRIPPER:
            gripper_joint = self.potentiometer
        return gripper_joint

    def get_controller_pose(self) -> Optional[np.ndarray]:
        """Get the pose of the controller.

        This method updates the tracker pose and converts it to the robot frame.
        It also applies a filter to the pose if required.

        Returns:
            Optional[np.ndarray]: The pose of the controller in the robot frame.
        """
        self.tracker.update_tracker_pose()
        pose = self.tracker.tracker_pose

        if self.is_filtered:
            pose = self.pose_filter.update(pose)
        pose = self.convert_pose(pose)
        self.former_pose = pose
        return pose

    def convert_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert the pose from the tracker to the robot frame.

        This method applies a transformation to the pose based on the tracker type
        and the arm being controlled.

        Args:
            pose (np.ndarray): The pose to be converted.
        Returns:
            np.ndarray: The converted pose.
        """
        if self.tracker_init_pose is None:
            return pose

        # Get the relative pose between the initial and the current pose
        relative_pose = np.linalg.inv(self.tracker_init_pose) @ pose

        if self.tracker_type == TrackerType.VIVE:
            relative_pose = self.convert_for_vive_tracker(relative_pose)

        elif self.tracker_type == TrackerType.ARUCO:
            T_cam_to_reachy = np.array(
                [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
            )
            relative_pose = T_cam_to_reachy @ relative_pose

        return relative_pose

    def convert_for_vive_tracker(self, pose: np.ndarray) -> np.ndarray:
        """Convert the pose for the Vive tracker.

        This method applies a transformation to the pose based on the arm being controlled.

        Args:
            pose (np.ndarray): The pose to be converted.
        Returns:
            np.ndarray: The converted pose.
        """
        if self.arm == "r_arm":
            angle_rotation = -30
        else:
            angle_rotation = 30

        rotation = R.from_euler("xyz", [0, 0, angle_rotation], degrees=True).as_matrix()
        Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
        pose = Trot @ pose

        if self.arm == "r_arm":
            rotation = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
            Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
            pose = Trot @ pose

        if self.arm == "l_arm":
            rotation = R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
            Trot = make_homogenous_matrix_from_rotation_matrix(rotation, [0, 0, 0])
            pose = Trot @ pose

        return pose

    def stop(self) -> None:
        """Stop the controller.

        This method stops the tracker and closes the Arduino connection.
        """
        self.stop_flag = True
        self.arduino.close()
        self.tracker.stop()
