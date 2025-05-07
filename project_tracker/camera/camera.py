from abc import ABC, abstractmethod
from collections import deque


class Camera(ABC):
    """
    Abstract class for camera.
    """

    def __init__(self):
        """Initialize the camera."""
        self.color_frame = deque(maxlen=1)

    @abstractmethod
    def get_frame(self):
        """Get the current frame from the camera."""
        pass
