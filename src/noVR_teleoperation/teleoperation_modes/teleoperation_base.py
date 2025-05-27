import time
from abc import ABC, abstractmethod

import numpy as np
from robots.reachy import Reachy2  # type: ignore
from utils import load_config  # type: ignore

DUAL_ARM = "dual_arm"
LEFT_ARM = "l_arm"
RIGHT_ARM = "r_arm"


class Teleoperation(ABC):
    """Base class for teleoperation with Reachy2 robot."""

    def __init__(self) -> None:
        """Initialize the teleoperation class.

        Loads the configuration from a YAML file and initializes the robot and controllers.
        """
        config = load_config("config.yaml")
        robot_ip = config.get("robot_ip", "localhost")
        mirror_mode = config.get("mirror_mode", False)
        self.control_mode = config.get("control_mode", "dual_arm")

        self.robot = Reachy2(robot_ip, mirror_mode)
        self.controllers: dict = {}
        self.controller_previous_pose = {LEFT_ARM: np.eye(4), RIGHT_ARM: np.eye(4)}
        self.robot_previous_pose = {LEFT_ARM: np.eye(4), RIGHT_ARM: np.eye(4)}

    @abstractmethod
    def init_teleoperation(self) -> None:
        pass

    def teleoperation(self) -> None:
        """Main loop for teleoperation.

        Initializes the teleoperation and enters a loop that calls the `step()` method at a fixed frequency.
        """
        try:
            self.init_teleoperation()
            frequency = 100
            while True:
                t = time.time()
                self.step()
                time.sleep(max(0, 1 / frequency - (time.time() - t)))
        except KeyboardInterrupt:
            print("Teleoperation stopped by user.")
            self.robot.stop()
            first_controller = next(iter(self.controllers.values()), None)
            if first_controller:
                first_controller.stop()

    @abstractmethod
    def step(self):
        pass
