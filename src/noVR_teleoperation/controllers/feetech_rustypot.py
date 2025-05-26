import time


class Feetech:
    """Feetech controller for servo motors

    It is used for the custom controller if the gripper is motorized and SO100.
    """

    def __init__(self, port, ids):
        """Initializes the Feetech controller with the specified serial port and servo IDs.
        Args:
            port (str): The serial port to which the Feetech controller is connected.
            ids (list[int]): List of servo IDs to control.
        """
        from rustypot.servo import Sts3215SyncController  # type: ignore

        self.io = Sts3215SyncController(
            serial_port=port,
            baudrate=1000000,
            timeout=0.1,
        )
        self.ids = ids

        self.io.write_mode(self.ids, [0] * len(self.ids))
        self.io.write_torque_enable(self.ids, [True] * len(self.ids))

    def stop(self) -> None:
        """Stops the Feetech controller by disabling torque on all servos."""
        self.io.write_torque_enable(self.ids, [False] * len(self.ids))

    def get_joints(self) -> list[float]:
        """Reads the current positions of the servos.
        Returns:
            list[float]: List of current positions of the servos in radians.
        """
        return self.io.read_present_position(self.ids)

    def set_joints(self, positions: list[float]) -> None:
        """Sets the target positions for the servos.
        Args:
            positions (list[float]): List of target positions for the servos in radians.
        """
        self.io.write_goal_position(self.ids, positions)

    def disable_torque(self) -> None:
        """Disables torque on all servos."""
        self.io.write_torque_enable(self.ids, [False] * len(self.ids))

    def enable_torque(self) -> None:
        """Enables torque on all servos."""
        self.io.write_torque_enable(self.ids, [True] * len(self.ids))

    def set_torque_limit(self, torque: float) -> None:
        """Sets the torque limit for all servos.
        Args:
            torque (float): The torque limit to set for the servos.
        """
        self.io.write_torque_limit(self.ids, [torque] * len(self.ids))

    def goto_joints(self, joints: list[float], duration: float) -> None:
        """Moves the servos to the specified joint positions over a given duration.
        Args:
            joints (list[float]): List of target joint positions in radians.
            duration (float): Duration over which to move the servos in seconds.
        """
        freq = 100
        steps = int(duration * freq)

        current_positions = self.get_joints()
        for i in range(steps):
            self.set_joints(
                [current_positions[j] + (joints[j] - current_positions[j]) * i / steps for j in range(len(joints))]
            )
            time.sleep(1 / freq)
