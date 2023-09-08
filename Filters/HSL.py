from numpy import ndarray
from Filters.Filter import Filter
import cv2
import numpy as np

class HSL(Filter):
    """
    Adds random hue saturation and lightness
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float64)

        image *= (np.random.rand((3)) / 3) + (1 - (1/6))

        return cv2.cvtColor(image.clip(0, 255).round().astype(np.uint8), cv2.COLOR_HLS2RGB)
    
class Hue(Filter):
    """
    Adds random hue 
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float64)

        image[:, :, 0] *= (np.random.rand((1)) / 3) + (1 - (1/6))

        return cv2.cvtColor(image.clip(0, 255).round().astype(np.uint8), cv2.COLOR_HLS2RGB)
    
class Saturation(Filter):
    """
    Adds random saturation 
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float64)

        image[:, :, 1] *= (np.random.rand((1)) / 3) + (1 - (1/6))

        return cv2.cvtColor(image.clip(0, 255).round().astype(np.uint8), cv2.COLOR_HLS2RGB)
    
class Lightness(Filter):
    """
    Adds random lightness 
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float64)

        image[:, :, 2] *= (np.random.rand((1)) / 3) + (1 - (1/6))

        return cv2.cvtColor(image.clip(0, 255).round().astype(np.uint8), cv2.COLOR_HLS2RGB)