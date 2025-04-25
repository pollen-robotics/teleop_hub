from abc import ABC, abstractmethod

from trackers.tracker import Tracker  # type: ignore


class Controller(ABC):
    def __init__(self) -> None:
        self.tracker: Tracker

    @abstractmethod
    def get_controller_pose(self):
        pass

    @abstractmethod
    def convert_pose(self, pose):
        pass
