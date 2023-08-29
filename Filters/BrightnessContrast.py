from Filters.Filter import *

class Brightness(Filter):
    """
    Adds Random Brightness to Image
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        return cv2.convertScaleAbs(image, alpha=1.0, beta=(self.rand.random()*60)-30)
    

class Contrast(Filter):
    """
    Adds Random Contrast to Image
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        return cv2.convertScaleAbs(image, beta=0.0, alpha=(self.rand.random() + 0.5))
    

class BrightnessContrast(Filter):
    """
    Adds Random Brightness and Contrast to Image
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        return cv2.convertScaleAbs(image, beta=(self.rand.random()*60)-30, alpha=(self.rand.random() + 0.5))  