import math
import sys
import time

import numpy
import pysurvive


# A helper function to update a single line of text
def update_text(txt):
    sys.stdout.write("\r" + txt)
    sys.stdout.flush()


# Convert a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in radians.
def quaternion_to_euler(q):
    w, x, y, z = q  # assuming pysurvive returns quaternion in (w, x, y, z)
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return numpy.array([roll, pitch, yaw])


# A simple buffer class to store pose samples over time.
class pose_sample_buffer:
    def __init__(self):
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.euler = []  # Euler angles (roll, pitch, yaw)
        self.quaternion = []  # Quaternion (w, x, y, z)

    def append(self, pos, quat, t):
        self.time.append(t)
        self.x.append(pos[0])
        self.y.append(pos[1])
        self.z.append(pos[2])
        self.quaternion.append(quat)
        self.euler.append(quaternion_to_euler(quat))


# A wrapper around a pysurvive object that provides methods to sample and retrieve poses.
class vr_tracked_device:
    def __init__(self, survive_obj):
        self.obj = survive_obj

    def get_name(self):
        return self.obj.Name()

    def get_pose(self):
        # pysurvive SimpleContext returns each object's pose as a tuple: (position, quaternion)
        return self.obj.Pose()

    def sample(self, num_samples, sample_rate):
        interval = 1 / sample_rate
        buf = pose_sample_buffer()
        sample_start = time.time()
        for _ in range(num_samples):
            start = time.time()
            pos, quat = self.get_pose()
            buf.append(pos, quat, time.time() - sample_start)
            sleep_time = interval - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
        return buf


# The main manager class that initializes pysurvive and organizes all tracked devices.
class triad_survive:
    def __init__(self, args=None):
        # Initialize the SimpleContext from pysurvive.
        if args is None:
            args = []
        self.ctx = pysurvive.SimpleContext(args)
        # Create a dictionary to store devices keyed by their name.
        self.devices = {}
        # Organize devices by type; here we use basic name matching.
        self.device_types = {"HMD": [], "Controller": [], "Tracker": []}
        for obj in self.ctx.Objects():
            name = obj.Name()
            lower = name.lower()
            if "tracker" in lower:
                self.device_types["Tracker"].append(name)
            elif "controller" in lower:
                self.device_types["Controller"].append(name)
            elif "hmd" in lower:
                self.device_types["HMD"].append(name)
            else:
                # If a device doesn't match any category, store it under its raw name.
                self.device_types.setdefault("Other", []).append(name)
            self.devices[name] = vr_tracked_device(obj)

    def rename_device(self, old_name, new_name):
        if old_name in self.devices:
            self.devices[new_name] = self.devices.pop(old_name)
            for key in self.device_types:
                for idx, dev_name in enumerate(self.device_types[key]):
                    if dev_name == old_name:
                        self.device_types[key][idx] = new_name
        else:
            print("Device", old_name, "not found.")

    def get_device_count(self):
        return len(self.devices)

    def list_devices(self):
        print("Discovered devices:")
        for dtype, names in self.device_types.items():
            print(f"  {dtype}s: {len(names)}")
            for name in names:
                dev = self.devices.get(name)
                if dev:
                    pos, quat = dev.get_pose()
                    euler = quaternion_to_euler(quat)
                    print(f"    {name}:")
                    print(f"      Pose: Pos = {pos}")
                    print(f"      Quaternion = {quat}")
                    print(f"      Euler (rad) = {euler}")
                else:
                    print(f"    {name}: (device not found)")

    def run(self):
        # An example loop to print out poses continuously.
        try:
            while self.ctx.Running():
                for name, dev in self.devices.items():
                    pos, quat = dev.get_pose()
                    print(f"{name}: Pos = {pos}, Quat = {quat}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting run loop.")


# Example usage when this script is run as main.
if __name__ == "__main__":
    # Initialize our triad_survive manager.
    triad = triad_survive()
    print("Device count:", triad.get_device_count())
    triad.list_devices()

    # For each device, take a short sample (e.g., 10 samples at 30 Hz)
    for name, dev in triad.devices.items():
        print(f"\nSampling from {name}...")
        buf = dev.sample(10, 30)
        for t, pos, quat, euler in zip(buf.time, buf.x, buf.y, buf.z):
            # For simplicity, print the timestamp and position.
            # (You can add printing of quaternion or Euler angles as needed.)
            print(
                f"Time: {t:.3f} sec, Position: ({pos:.3f}, {buf.y[buf.time.index(t)]:.3f}, {buf.z[buf.time.index(t)]:.3f})"
            )
        print("-----")

    # Optionally, you can run a continuous loop:
    # triad.run()
