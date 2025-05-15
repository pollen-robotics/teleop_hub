import time
from typing import Optional

FEETECH_GRIPPER = "feetech"
POTENTIOMETER_GRIPPER = "potentiometer"


class ArduinoController:
    """
    This class handles the communication with the Arduino board.
    It reads the joystick input and button states from the Arduino.
    """

    def __init__(self, port: str, gripper_type: str) -> None:
        """Initializes the ArduinoController with the specified serial port.

        Args :
            port (str): The serial port to which the Arduino is connected.
        """
        import serial  # type: ignore

        self.ser = serial.Serial(port, 9600)
        self.ser.flushInput()
        self.gripper_type = gripper_type

    def read(
        self,
    ) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Reads the joystick input and button states from the Arduino.

        Returns:
            tuple: A tuple containing the x and y coordinates of the joystick,
                   the button command, and the states of button A and button B.
        """
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode("utf-8", errors="ignore").strip()
            values = data.split(",")
            x_input = int(values[0])
            y_input = int(values[1])
            button_cmd = int(values[2])
            buttonA = int(values[3])
            buttonB = int(values[4])
            if self.gripper_type == FEETECH_GRIPPER:
                return x_input, y_input, button_cmd, buttonA, buttonB, None
            else:
                potentiometer = int(values[5])
                return x_input, y_input, button_cmd, buttonA, buttonB, potentiometer

        if self.gripper_type == FEETECH_GRIPPER:
            return None, None, None, None, None, None
        return None, None, None, None, None, None

    def close(self) -> None:
        """Closes the serial connection to the Arduino."""
        self.ser.close()


if __name__ == "__main__":
    gripper_type = POTENTIOMETER_GRIPPER  # Change to FEETECH_GRIPPER for the other gripper
    # arduino = ArduinoController("/dev/noVR_right_arduino", gripper_type)
    arduino = ArduinoController("/dev/noVR_left_arduino", gripper_type)

    while True:
        if gripper_type == FEETECH_GRIPPER:
            x, y, button_cmd, buttonA, buttonB, _ = arduino.read()
        else:
            x, y, button_cmd, buttonA, buttonB, potentiometer = arduino.read()
        if x is not None:
            if gripper_type == FEETECH_GRIPPER:
                print(f"x: {x}, y: {y}, button_cmd: {button_cmd}, buttonA: {buttonA}, buttonB: {buttonB}")
            else:
                print(
                    f"x: {x}, y: {y}, button_cmd: {button_cmd}, buttonA: {buttonA}, buttonB: {buttonB},"
                    + f"potentiometer: {potentiometer}",
                )
        else:
            print("No data")
        time.sleep(0.1)
