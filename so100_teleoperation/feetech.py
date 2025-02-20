from pypot.feetech import FeetechSTS3215IO
import time
import threading
import math
from lerobot.common.utils.kinematics import RobotKinematics



#create Class feetech
class Feetech:
    def __init__(self, port):


        self.io = FeetechSTS3215IO(
            port,
            baudrate=1000000,
            use_sync_read=True,
        )

        for i in range(1,7):
            self.io.disable_torque([2])

        time.sleep(1)
        # self.id = 1
        # self.io.enable_torque([self.id])
        # self.io.set_mode({1:0})
        # # self.io.set_goal_speed({1:0})
        # self.io.set_torque_limit({1:1000})
        # self.t = 0.01
        # self.pwm = 0
        # self.stop = False

    def set_pwm(self, pwm):
        self.pwm = pwm

    def angle_diff(self, angle1, angle2):
        """Returns the smallest distance between 2 angles"""
        d = angle1 - angle2
        d = ((d + math.pi) % (2 * math.pi)) - math.pi
        return d

    def run(self):
        print("Running")
        while True:
            self.io.enable_torque([1])
            time.sleep(self.t * self.pwm / 100)
            self.io.disable_torque([1])
            time.sleep(self.t * (100 - self.pwm) / 100)
            if self.stop:
                break


    def close(self):
        self.stop = True
        self.io.disable_torque([self.id])
        self.io.close([self.id])

    def get_position(self, id):
        return self.io.get_present_position([id])[0]
    
    def set_position(self, id, position):
        self.io.set_goal_position({id: position})
    
    def goto_position(self, id, position, duration):
        freq = 100
        steps = int(duration * freq)
        current_position = self.get_position(id)
        for i in range(steps):
            self.set_position(id, current_position + (position - current_position) * i / steps)
            time.sleep(1/freq)
        self.set_position(id, position)

if __name__ == "__main__":
    try: 
        feetech = Feetech("/dev/ttyACM0")
        time.sleep(1)
        
        pos = []
        for i in range(1,7):
            position = feetech.get_position(i)
            print(f"Id : {id} Position {i}: {position}")
            pos.append(position)

        # pos[3] += 180
        # pos[4] -= 90

        # pos[4] *= -1
        # # tip_pose = RobotKinematics.fk_gripper_tip(pos)
        # print(tip_pose)
            # feetech.set_position(i, 0)
            # time.sleep(1)
        
    except KeyboardInterrupt:
        feetech.close()
        print("Bye")



# try:
#     # for baud in [9600, 57600, 115200, 1000000, 2000000]:
#         # io = DxlIO('/dev/ttyACM0', baudrate=baud, timeout=0.1)
#         # print(f"Testing baudrate {baud}: {io.scan()}")
#     io = FeetechSTS3215IO(
#         "/dev/ttyACM0",
#         baudrate=1000000,
#         use_sync_read=True,
#     )
#     # print(f"Testing baudrate : {io.scan()}")

#     time.sleep(1)
#     id = 1

#     io.enable_torque([id])
#     time.sleep(1)
#     io.set_mode({1:1})
#     io.set_goal_speed({1:0})
#     io.set_torque_limit({1:1000})

#     # time.sleep(3)
#     t = 0.01
#     for i in range(0, 100):
#         pwm = i
#         print(pwm)
#         for j in range(0, 10):
#             io.enable_torque([1])
#             time.sleep(t * pwm / 100)
#             io.disable_torque([1])
#             time.sleep(t * (100 - pwm) / 100)
#     io.enable_torque([1])
#     time.sleep(3)
#     # print(io.get_present_position([id]))
#     # time.sleep(1)
#     io.disable_torque([id])
#     io.close()


# except KeyboardInterrupt:
#     io.disable_torque([id])
#     io.close()
#     print("Bye")
#     exit(0)

    