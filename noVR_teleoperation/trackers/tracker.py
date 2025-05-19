from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np


class TrackerType(Enum):
    """Enum for the type of tracker."""

    ARUCO = "aruco"
    VIVE = "vive"
    RGBD = "rgbd"
    SO_ARM = "so_arm"
    DUALSHOCK = "dualshock"


class Tracker(ABC):
    """Abstract class for tracking a target."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.tracker_type: Optional[TrackerType]
        self.tracker_pose: Optional[np.ndarray] = None

    @abstractmethod
    def update_tracker_pose(self) -> None:
        """Update the tracker pose."""
        pass
