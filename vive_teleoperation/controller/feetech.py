from pypot.feetech import FeetechSTS3215IO
import time
import threading
import math
# from lerobot.common.utils.kinematics import RobotKinematics


#create Class feetech
class Feetech:
    def __init__(self, port):


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
        self.disable_torque()
        self.io.close()

    def get_position(self, id):
        return self.io.get_present_position([id])[0]
    
    def set_position(self, id, position):
        self.io.set_goal_position({id: position})

    def disable_torque(self):
        for id in self.ids:
            self.io.disable_torque([id])

    def enable_torque(self):
        for id in self.ids:
            self.io.enable_torque([id])
    
    def goto_position(self, id, position, duration):
        freq = 100
        steps = int(duration * freq)
        current_position = self.get_position(id)
        for i in range(steps):
            self.set_position(id, current_position + (position - current_position) * i / steps)
            time.sleep(1/freq)
        self.set_position(id, position)

    def goto_joints(self, joints, duration):
        freq = 100
        steps = int(duration * freq)
        current_positions = self.get_joints()
        for i in range(steps):
            for j in range(1, len(self.ids) +1):
                self.set_position(j, current_positions[j-1] + (joints[j-1] - current_positions[j-1]) * i / steps)
            time.sleep(1/freq)
        for j in range(1, len(self.ids) +1):
            self.set_position(j, joints[j-1])

    def get_joints(self):
        # print("ici")
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


    