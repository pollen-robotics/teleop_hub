from pypot.feetech import FeetechSTS3215IO
import time
import threading
import math
# from lerobot.common.utils.kinematics import RobotKinematics


#create Class feetech
class Feetech:
    def __init__(self, port, nb_dof=5):


        self.io = FeetechSTS3215IO(
            port,
            baudrate=1000000,
            use_sync_read=True,
        )

        if nb_dof == 5:
            self.ids = [1, 2, 3, 4, 5, 6]
        elif nb_dof == 6:
            self.ids = [1, 2, 3, 4, 5, 6, 7]
        else:
            raise ValueError("nb_dof must be 5 or 6")

        for id in self.ids:
            self.io.set_mode({id: 0})

        self.disable_torque()

        # time.sleep(1)
        # self.id = 1
        # self.io.enable_torque([self.id])
        # self.io.set_mode({1:0})
        # # self.io.set_goal_speed({1:0})
        # self.io.set_torque_limit({1:1000})
        self.t = 0.01
        # self.pwm = 0
        self.stop = False

    def set_torque_limit(self, torque):
        for id in self.ids:
            self.io.set_torque_limit({id: torque})

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
            print(self.pwm)
            if self.pwm == 0:
                for id in self.ids:
                    self.io.disable_torque([id])
                time.sleep(self.t)
            elif self.pwm == 100:
                for id in self.ids:
                    self.io.enable_torque([id])
                time.sleep(self.t)
            else:
                for id in self.ids:
                    self.io.enable_torque([id])
                time.sleep(self.t * self.pwm / 100)
                for id in self.ids:
                    self.io.disable_torque([id])
                time.sleep(self.t * (100 - self.pwm) / 100)
            if self.stop:
                break


    def close(self):
        self.stop = True
        self.disable_torque()
        self.io.close()

    def get_position(self, id):
        return self.io.get_present_position([id])[0]
    
    def set_position(self, id, position):
        self.io.set_goal_position({id: position})

    def set_joints(self, joints):
        for i in range(len(joints)):
            self.set_position(i + 1, joints[i])

    def disable_torque(self):
        for id in self.ids:
            self.io.disable_torque([id])

    def enable_torque(self):
        for id in self.ids:
            self.io.enable_torque([id])

    def set_torque_limit(self, torque):
        for id in self.ids:
            self.io.set_torque_limit({id: torque})
    
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
        print(joints)
        print(current_positions)
        for i in range(steps):
            for j in range(len(joints)):
                self.set_position(j + 1, current_positions[j] + (joints[j] - current_positions[j]) * i / steps)
            time.sleep(1/freq)
        for j in range(len(joints)):
            self.set_position(j+1, joints[j])

    def get_joints(self):
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


    