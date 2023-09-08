from Filters.Filter import *
from Filters.Filter import ndarray

class Blur(Filter):
    """
    Adds random Blur to the image
    """

    def __init__(self, min:int=3, max:int=10) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, image: ndarray) -> ndarray:
        k = self.rand.randint(self.min, self.max)
        if k%2 == 0:
            k+= 1
        return cv2.blur(image, (k, k))
    

class GaussianBlur(Blur):
    """
    Adds random Gaussian blur to the image
    """
    def __init__(self, min: int = 3, max: int = 10) -> None:
        super().__init__(min, max)
    
    def forward(self, image: ndarray) -> ndarray:
        k = self.rand.randint(self.min, self.max)
        if k%2 == 0:
            k+= 1
        return cv2.GaussianBlur(image, (k, k), 0)