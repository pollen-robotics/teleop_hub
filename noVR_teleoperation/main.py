from teleoperation_mode.teleoperation import Teleoperation  # type: ignore
from teleoperation_mode.teleoperation_rgbd import TeleoperationRGBD
from teleoperation_mode.teleoperation_so_arm import SoArmTeleoperation
from teleoperation_mode.teleoperation_with_joystick import \
    JoystickTeleoperation
from trackers.tracker import TrackerType  # type: ignore
from utils import load_config  # type: ignore

if __name__ == "__main__":
    config = load_config("config.yaml")
    tracker_type_str = config.get("tracker_type", TrackerType.ARUCO).upper()
    tracker_type = getattr(TrackerType, tracker_type_str, None)

    teleoperation: Teleoperation

    if tracker_type == TrackerType.ARUCO or tracker_type == TrackerType.VIVE:
        teleoperation = JoystickTeleoperation(tracker_type)
    elif tracker_type == TrackerType.RGBD:
        teleoperation = TeleoperationRGBD()
    elif tracker_type == TrackerType.SO_ARM:
        teleoperation = SoArmTeleoperation(config["port"])
    else:
        raise ValueError(f"Invalid tracker type: {tracker_type}")

    try:
        teleoperation.teleoperation()

    except KeyboardInterrupt:
        for controller in teleoperation.controllers.values():
            controller.stop()
        teleoperation.robot.stop()
