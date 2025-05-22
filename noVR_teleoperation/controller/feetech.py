import math
import time

# from pypot.feetech import FeetechSTS3215IO  # type: ignore

# from lerobot.common.utils.kinematics import RobotKinematics


class Feetech:
    """Feetech class for controlling the Feetech servo motors, used for the gripper commands.

    This class provides methods to set the PWM, get and set positions, and enable/disable torque.
    """

    def __init__(self, port: str):
        """Initialize the Feetech class.

        Args:
            port (str): The port to which the Feetech servo motors are connected.
        """
        self.io = FeetechSTS3215IO(
            port,
            baudrate=1000000,
            use_sync_read=True,
        )

        self.ids = [1]

        for id in self.ids:
            self.io.set_mode({id: 0})

        self.disable_torque()

        self.t = 0.01
        self.stop = False

    def set_pwm(self, pwm: int):
        """Set the PWM for the servo motors.
        Args:
            pwm (int): The PWM value to set.
        """
        self.pwm = pwm

    def angle_diff(self, angle1: float, angle2: float) -> float:
        """Returns the smallest distance between 2 angles"""
        d = angle1 - angle2
        d = ((d + math.pi) % (2 * math.pi)) - math.pi
        return d

    def run(self) -> None:
        """Run the servo motors with the specified PWM value.

        This method enables the torque for the servo motors and runs them in a loop.
        The loop will run until the stop flag is set to True.
        """
        print("Running")
        while True:
            self.io.enable_torque([1])
            time.sleep(self.t * self.pwm / 100)
            self.io.disable_torque([1])
            time.sleep(self.t * (100 - self.pwm) / 100)
            if self.stop:
                break

    def close(self) -> None:
        """Close the Feetech class.

        This method disables the torque for the servo motors and closes the IO connection.
        """
        self.disable_torque()
        self.io.close()

    def get_position(self, id: int) -> float:
        """Get the current position of the servo motor.

        Args:
            id (int): The ID of the servo motor.
        Returns:
            float: The current position of the servo motor.
        """
        return self.io.get_present_position([id])[0]

    def set_position(self, id: int, position: float) -> None:
        """Set the position of the servo motor.

        Args:
            id (int): The ID of the servo motor.
            position (float): The position to set for the servo motor.
        """
        self.io.set_goal_position({id: position})

    def disable_torque(self) -> None:
        """Disable the torque for the servo motors.

        This method disables the torque for all servo motors in the Feetech class.
        """
        for id in self.ids:
            self.io.disable_torque([id])

    def enable_torque(self) -> None:
        """Enable the torque for the servo motors.

        This method enables the torque for all servo motors in the Feetech class.
        """
        for id in self.ids:
            self.io.enable_torque([id])

    def goto_position(self, id: int, position: float, duration: float) -> None:
        """Move the servo motor to a specified position over a given duration.

        Args:
            id (int): The ID of the servo motor.
            position (float): The position to move to.
            duration (float): The duration over which to move.
        """
        freq = 100
        steps = int(duration * freq)
        current_position = self.get_position(id)
        for i in range(steps):
            self.set_position(
                id, current_position + (position - current_position) * i / steps
            )
            time.sleep(1 / freq)
        self.set_position(id, position)

    def goto_joints(self, joints: list, duration: float) -> None:
        """Move the servo motors to specified joint positions over a given duration.

        Args:
            joints (list): The joint positions to move to.
            duration (float): The duration over which to move.
        """
        freq = 100
        steps = int(duration * freq)
        current_positions = self.get_joints()
        for i in range(steps):
            for j in range(1, len(self.ids) + 1):
                self.set_position(
                    j,
                    current_positions[j - 1]
                    + (joints[j - 1] - current_positions[j - 1]) * i / steps,
                )
            time.sleep(1 / freq)
        for j in range(1, len(self.ids) + 1):
            self.set_position(j, joints[j - 1])

    def get_joints(self) -> list:
        """Get the current joint positions of the servo motors.

        Returns:
            list: The current joint positions of the servo motors.
        """
        return [self.get_position(id) for id in self.ids]


if __name__ == "__main__":
    try:
        feetech = Feetech("/dev/ttyACM0")
        time.sleep(1)
        pos = []
        for id in feetech.ids:
            position = feetech.get_position(id)
            print(f"Id : {id} Position : {position}")
            pos.append(position)

    except KeyboardInterrupt:
        feetech.close()
        print("Bye")
