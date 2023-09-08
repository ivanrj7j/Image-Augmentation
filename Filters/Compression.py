from numpy import ndarray
from Filters.Filter import Filter
import cv2

class JPEGCompression(Filter):
    def __init__(self, amount:int=5) -> None:
        super().__init__()
        self.amount = 10

    def forward(self, image: ndarray) -> ndarray:
        encoded_image, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.amount])

        return cv2.imdecode(buffer, 1)