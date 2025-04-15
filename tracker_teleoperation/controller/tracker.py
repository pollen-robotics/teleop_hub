from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np


class TrackerType(Enum):
    ARUCO = "aruco"
    VIVE = "vive"


class Tracker(ABC):
    """
    Abstract class for tracking a target.
    """
    def __init__(self, arm: str):
        self.arm = arm
        self.tracker_type: Optional[TrackerType]
        self.tracker_pose: Optional[np.ndarray] = None

    @abstractmethod
    def update_tracker_pose(self):
        """
        Update the tracker pose.
        """
        pass
