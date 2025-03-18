import serial
import time
import math
import numpy as np
import threading

class JoystickData:
    def __init__(self, port):
        self.ser = serial.Serial(port, 9600)
        time.sleep(2)

    def read(self):
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode('utf-8').strip()
            values = data.split(',')
            x_input = int(values[0])
            y_input = int(values[1])
            button_cmd = int(values[2])
            sw2 = int(values[3])
            sw3 = int(values[4])
            # print(f"VRX: {x_input}, VRY: {y_input}, SW: {button_cmd}", f"SW2: {sw2}, SW3: {sw3}")

            return x_input, y_input, button_cmd, sw2, sw3

        return None, None, None, None, None

class MobileBaseController:
    def __init__(self, reachy, port_joystick='/dev/ttyACM1', port_joystick_2='/dev/ttyACM2',  two_trackers_mode=False, ): #voir les ports
        self.mobile_base = reachy.mobile_base
        self.two_trackers_mode = two_trackers_mode

        if not two_trackers_mode:
            self.joystick_data = JoystickData(port_joystick)
            self.translation_mode = True
            self.rotation_mode = False
            self.button_ready = True

        else:
            self.joystick_data_translation = JoystickData(port_joystick)
            self.joystick_data_rotation = JoystickData(port_joystick_2)

        self.command_getter = threading.Thread(target=self.run)
        self.command_getter.start()

    def change_mode(self):
        self.translation_mode = not self.translation_mode
        self.rotation_mode = not self.rotation_mode

    def get_button_command(self, button_cmd):
        if button_cmd == 1:
            self.button_ready = True
        elif self.button_ready and button_cmd == 0:
            self.change_mode()
            self.button_ready = False
            print(
                f"button pressed : change mode to translation : {self.translation_mode}, rotation : {self.rotation_mode}"
            )

    def convert_xy_command(self, x_input, y_input):
        command_max = 0.4
        center_joystick = 512
        x_goal, y_goal = 0, 0

        if not np.isclose(x_input, center_joystick, atol=15):
            y_goal = command_max / center_joystick * x_input - command_max
        if not np.isclose(y_input, center_joystick, atol=15):
            x_goal = command_max / center_joystick * y_input - command_max

        return x_goal, y_goal

    def send_command(self, x1_input, y1_input, x2_input=0, y2_input=0):
        x_goal, y_goal, theta = 0, 0, 0
        rotation_threshold = 300
        center = [1023 // 2, 1023 // 2]

        if self.two_trackers_mode:
            # get the translation from the first joystick
            x_goal, y_goal = self.convert_xy_command(x1_input, y1_input)

            # and the rotation from the second joystick
            vector = [x2_input - center[0], y2_input - center[1]]
            norm = math.sqrt(vector[0]**2 + vector[1]**2)
            if norm > rotation_threshold:
                angle = math.atan2(vector[0], vector[1])
                theta = np.rad2deg(angle)
                theta /= 2

        else:
            # get either translation or rotation from the joystick
            if self.translation_mode:
                x_goal, y_goal = self.convert_xy_command(x1_input, y1_input)
                self.mobile_base.set_goal_speed(vx=x_goal, vy=y_goal, vtheta=0)

            elif self.rotation_mode:
                vector = [x1_input - center[0], y1_input - center[1]]
                norm = math.sqrt(vector[0]**2 + vector[1]**2)
                if norm > rotation_threshold:
                    angle = math.atan2(vector[0], vector[1])
                    theta = np.rad2deg(angle)
                    theta /= 2

        print(f"goal : {x_goal}, {y_goal}, {theta}")
        if np.any([x_goal, y_goal, theta]):
            
            self.mobile_base.set_goal_speed(x=x_goal, y=y_goal, theta=theta)
            self.mobile_base.send_speed_command()

    def run(self):
        data = [0, 0, 0, 0, 0]
        while True:
            if self.two_trackers_mode:
                # print("hey")
                x1_input, y1_input, button_joystick, sw2, sw3 = self.joystick_data_translation.read()
                x2_input, y2_input, button_joystick_2, sw2_2, sw3_2 = self.joystick_data_rotation.read()
                # print(x1_input, y1_input, x2_input, y2_input)
                if x1_input is not None:
                    data[0] = x1_input
                if y1_input is not None:
                    data[1] = y1_input
                if x2_input is not None:
                    data[2] = x2_input
                if y2_input is not None:
                    data[3] = y2_input
                if button_joystick is not None:
                    data[4] = button_joystick
                # if x1_input is None or x2_input is None:
                #     continue
                self.send_command(data[0], data[1], data[2], data[3])
            else:
                x_input, y_input, button_joystick, buttonA, buttonB = self.joystick_data.read()
                if x_input is None:
                    continue
                self.get_button_command(button_joystick)
                self.send_command(x_input, y_input)


# if __name__ == "__main__" :
#     joystick_data_translation = JoystickData("/dev/noVR_left_arduino")
#     joystick_data_rotation = JoystickData("/dev/noVR_right_arduino")

#     while True:

#         data= joystick_data_translation.read()
#         if data is not None:
#             # print(f"aaaaa {data}")

#         else:
#             print("None")
#         data = joystick_data_rotation.read()
#         if data is not None:
#             print(f"bbbb {data}")
#         else:
#             print("None")

