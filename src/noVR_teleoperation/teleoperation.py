from teleoperation_modes.teleoperation_base import Teleoperation
from teleoperation_modes.teleoperation_custom_controller import (
    CustomControllerTeleoperation,
)
from teleoperation_modes.teleoperation_gamepad import GamepadTeleoperation
from teleoperation_modes.teleoperation_rgbd import TeleoperationRGBD
from teleoperation_modes.teleoperation_so_arm import SoArmTeleoperation
from trackers.tracker import TrackerType  # type: ignore
from utils import load_config  # type: ignore

if __name__ == "__main__":
    config = load_config("config.yaml")
    tracker_type_str = config.get("tracker_type", TrackerType.ARUCO).upper()
    print(f"Tracker type from config: {tracker_type_str}")
    tracker_type = getattr(TrackerType, tracker_type_str, None)
    print(f"Using tracker type: {tracker_type}")

    teleoperation: Teleoperation

    if tracker_type == TrackerType.ARUCO or tracker_type == TrackerType.VIVE:
        teleoperation = CustomControllerTeleoperation(tracker_type)
    elif tracker_type == TrackerType.RGBD:
        teleoperation = TeleoperationRGBD()
    elif tracker_type == TrackerType.ARM:
        teleoperation = SoArmTeleoperation(config["port"])
    elif tracker_type == TrackerType.GAMEPAD:
        teleoperation = GamepadTeleoperation(config["angle_step"], config["x_y_joystick_ratio"], config["z_increment"])
    else:
        raise ValueError(f"Invalid tracker type: {tracker_type}")

    teleoperation.teleoperation()
