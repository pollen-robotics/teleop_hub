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
            buttonA = int(values[3])
            buttonB = int(values[4])
            
            return x_input, y_input, button_cmd, buttonA, buttonB
        
        return None, None, None, None, None


class MobileBaseController:
    def __init__(self, reachy, two_trackers_mode=False, left_port_joystick='/dev/ttyACM1', right_port_joystick='/dev/ttyACM2'): #voir les ports
        self.reachy = reachy
        self.mobile_base = reachy.mobile_base
        self.two_trackers_mode = two_trackers_mode

        if not two_trackers_mode:
            self.joystick_data = JoystickData(left_port_joystick)
            self.translation_mode = True
            self.rotation_mode = False
            self.button_ready = True

        else:
            self.joystick_data_translation = JoystickData(left_port_joystick)
            self.joystick_data_rotation = JoystickData(right_port_joystick)
            self.mobile_base_mode = True
            self.antenna_mode = False
            self.button_ready = True
            self.stop = False


    def change_mode_head(self):
        self.mobile_base_mode = not self.mobile_base_mode
        self.antenna_mode = not self.antenna_mode

    def get_button_command_head(self, button_cmd):
        if button_cmd == 1:
            self.button_ready = True
        elif self.button_ready and button_cmd == 0:
            self.change_mode_head()
            self.button_ready = False
            print(
                f"button pressed : change mode to mobile base : {self.mobile_base_mode}, head : {self.antenna_mode}"
            )

    # def change_mode(self):
    #     self.translation_mode = not self.translation_mode
    #     self.rotation_mode = not self.rotation_mode

    # def get_button_command(self, button_cmd):
    #     if button_cmd == 1:
    #         self.button_ready = True
    #     elif self.button_ready and button_cmd == 0:
    #         self.change_mode()
    #         self.button_ready = False
    #         print(
    #             f"button pressed : change mode to translation : {self.translation_mode}, rotation : {self.rotation_mode}"
    #         )

    def convert_xy_command(self, x_input, y_input):
        command_max = 0.4
        center_joystick = 512
        x_goal, y_goal = 0, 0

        if not np.isclose(x_input, center_joystick, atol=15):
            y_goal = command_max / center_joystick * x_input - command_max
        if not np.isclose(y_input, center_joystick, atol=15):
            x_goal = command_max / center_joystick * y_input - command_max

        return x_goal, y_goal

    def convert_antenna_control(self, input):
        center_joystick = 512
        command_max = 20
        delta = command_max/ center_joystick * input - command_max
        print(delta)
        present_position = self.reachy.head.r_antenna.present_position
        print(present_position)
        command = present_position + delta
        print(command)
        return command


    def send_command(self, x1_input, y1_input, x2_input=0, y2_input=0):
        x_goal, y_goal, theta = 0, 0, 0
        rotation_threshold = 300
        center = [1023 // 2, 1023 // 2]

        if self.two_trackers_mode:
            if self.mobile_base_mode:
                # get the translation from the first joystick
                x_goal, y_goal = self.convert_xy_command(x1_input, y1_input)

                # and the rotation from the second joystick
                vector = [x2_input - center[0], y2_input - center[1]]
                norm = math.sqrt(vector[0]**2 + vector[1]**2)
                if norm > rotation_threshold:
                    angle = math.atan2(vector[0], vector[1])
                    theta = np.rad2deg(angle)
                    theta /= 2
                self.mobile_base.set_goal_speed(x=-x_goal, y=-y_goal, theta=theta)
                self.mobile_base.send_speed_command()

            elif self.antenna_mode:
                x_goal, y_goal = self.convert_xy_command(x1_input, y1_input)

                antenna_opening = self.convert_antenna_control(x2_input)
                self.reachy.head.r_antenna.goal_position = antenna_opening
                self.reachy.head.l_antenna.goal_position = -antenna_opening
                self.reachy.send_goal_positions()

                self.mobile_base.set_goal_speed(x=-x_goal, y=-y_goal, theta=0)
                self.mobile_base.send_speed_command()

        else:
            # get either translation or rotation from the joystick
            if self.translation_mode:
                x_goal, y_goal = self.convert_xy_command(x1_input, y1_input)
                self.mobile_base.set_goal_speed(x=x_goal, y=y_goal, theta=0)

            elif self.rotation_mode:
                vector = [x1_input - center[0], y1_input - center[1]]
                norm = math.sqrt(vector[0]**2 + vector[1]**2)
                if norm > rotation_threshold:
                    angle = math.atan2(vector[0], vector[1])
                    theta = np.rad2deg(angle)
                    theta /= 2
            if np.any([x_goal, y_goal, theta]):
                # print(f"goal : {x_goal}, {y_goal}, {theta}")
                self.mobile_base.set_goal_speed(x=x_goal, y=y_goal, theta=theta)
                self.mobile_base.send_speed_command()


    def run(self):
        data = [0, 0, 0, 0, 0]
        while True:
            if self.two_trackers_mode:
                x1_input, y1_input, button_joystick, buttonA, buttonB = self.joystick_data_translation.read()
                x2_input, y2_input, button_joystick_2, buttonA_2, buttonB_2 = self.joystick_data_rotation.read()
                # print("hey")
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
                if buttonA_2 is not None:
                    self.get_button_command_head(buttonA_2)
                self.send_command(data[0], data[1], data[2], data[3])

                if buttonA is not None:
                    self.stop = (buttonA == 0)
            else:
                x_input, y_input, button_joystick, buttonA, buttonB = self.joystick_data.read()
                if x_input is None:
                    continue
                self.get_button_command(button_joystick)
                self.send_command(x_input, y_input)
            time.sleep(0.001)


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

