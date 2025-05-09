import serial  # type: ignore
import time


class ArduinoController:
    def __init__(self, port):
        self.ser = serial.Serial(port, 9600)
        self.ser.flushInput()

    def read(self):
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode("utf-8", errors="ignore").strip()
            values = data.split(",")
            x_input = int(values[0])
            y_input = int(values[1])
            button_cmd = int(values[2])
            buttonA = int(values[3])
            buttonB = int(values[4])
            potentiometer = int(values[5])

            return x_input, y_input, button_cmd, buttonA, buttonB, potentiometer

        return None, None, None, None, None, None

    def close(self):
        self.ser.close()


if __name__ == "__main__":
    arduino = ArduinoController("/dev/noVR_right_arduino")
    while True:
        x, y, button_cmd, buttonA, buttonB, potentiometer = arduino.read()
        if x is not None:
            print(
                f"x: {x}, y: {y}, button_cmd: {button_cmd}, buttonA: {buttonA}, buttonB: {buttonB}, potentiometer: {potentiometer}"
            )
        else:
            print("No data")
        time.sleep(0.1)
