from collections import deque

import numpy as np


class MedianFilter:
    """A simple median filter for smoothing data.

    This filter maintains a sliding window of the last N measurements
    and returns the median of those measurements.
    """

    def __init__(self, filter_size=5):
        self.filter_size = filter_size
        self.measurements = deque(maxlen=filter_size)

    def update(self, measurement):
        self.measurements.append(measurement)
        if len(self.measurements) == self.filter_size:
            return np.median(np.array(self.measurements), axis=0)
        else:
            return measurement
