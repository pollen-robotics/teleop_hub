from rustypot.servo import Sts3215SyncController
import time
import math

# from lerobot.common.utils.kinematics import RobotKinematics


# Create Class feetech
class FeetechSOArm:
    def __init__(self, port, ids):

        # self.io = Sts3215SyncController(serial_port='/dev/ttyACM0', baudrate=100000, timeout=0.1)

        self.io = Sts3215SyncController(
        serial_port='/dev/ttyACM0', 
        baudrate=1000000, 
        timeout=0.1,
    )
        self.ids = ids

        self.io.write_mode(self.ids, [0] * len(self.ids))
        self.io.write_torque_enable(self.ids, [True] * len(self.ids))

    def stop(self):
        self.io.write_torque_enable(self.ids, [False] * len(self.ids))

    def get_joints(self):
        return self.io.read_present_position(self.ids)
    
    def set_joints(self, positions):
        self.io.write_goal_position(self.ids, positions)

    def disable_torque(self):
        self.io.write_torque_enable(self.ids, [False] * len(self.ids))

    def enable_torque(self):
        self.io.write_torque_enable(self.ids, [True] * len(self.ids))

    def set_torque_limit(self, torque):
        self.io.write_torque_limit(self.ids, [torque] * len(self.ids))

    def goto_joints (self, joints, duration):
        freq = 100
        steps = int(duration * freq)

        current_positions = self.get_joints()
        for i in range(steps):
            self.set_joints(
                [
                    current_positions[j]
                    + (joints[j] - current_positions[j]) * i / steps
                    for j in range(len(joints))
                ]
            )
            time.sleep(1 / freq)


if __name__ == "__main__":
    try:
        feetech = FeetechSOArm("/dev/ttyACM0", [1,2,3,4,5,6])

        while True:
            time.sleep(1)
            # print(feetech.io.read_present_position([1]))*
            feetech.goto_joints([0, 0, 0, 0, 0, 0], 1)
            time.sleep(1)

        # feetech.get_positions()
    except KeyboardInterrupt:
        feetech.stop()
        print("Bye")
