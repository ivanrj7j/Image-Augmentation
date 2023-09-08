from numpy import ndarray
import numpy as np
from Filters import Filter
import cv2

class Noise(Filter):
    """
    Adds noise to image
    """
    def __init__(self, mean:int=0, stdDeviation:int=3) -> None:
        super().__init__()
        self.mean = mean
        self.stdDeviation = stdDeviation

    def forward(self, image: ndarray) -> ndarray:
        noise = np.random.normal(self.mean, self.stdDeviation, image.shape).astype(np.uint8)
        return cv2.add(image, noise)