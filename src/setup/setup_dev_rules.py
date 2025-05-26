import os
import subprocess


def install_udev_rules():
    current_dir = os.path.dirname(__file__)
    rules_file = os.path.abspath(os.path.join(current_dir, "11-noVR.setup.rules"))
    target_path = "/etc/udev/rules.d/11-noVR.setup.rules"

    print(f"Installing udev rules from {rules_file} to {target_path}...")

    try:
        subprocess.run(["sudo", "cp", rules_file, target_path], check=True)
        subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=True)
        subprocess.run(["sudo", "udevadm", "trigger"], check=True)
        print("Udev rules installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing udev rules: {e}")
