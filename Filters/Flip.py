from Filters.Filter import *


class HorizontalFlip(Filter):
    """
    Flips the image in X axis
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        return cv2.flip(image, 0)
    
    def forwardWithBBox(self, image: ndarray, bBoxes: list[COCO]):
        flippedImage = self.forward(image)
        height, width = image.shape[:2]
        newBBoxes = []

        for bBox in bBoxes:
            newBBoxes.append(COCO(bBox.points['x'], height-bBox.points['y']-bBox.points['height'], bBox.points['width'], bBox.points['height']))

        return flippedImage, newBBoxes
    

class VerticalFlip(Filter):
    """
    Flips the image in Y axis
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        return cv2.flip(image, 1)
    
    def forwardWithBBox(self, image: ndarray, bBoxes: list[COCO]):
        flippedImage = self.forward(image)
        height, width = image.shape[:2]
        newBBoxes = []

        for bBox in bBoxes:
            newBBoxes.append(COCO(width-bBox.points['x']-bBox.points['width'], bBox.points['y'], bBox.points['width'], bBox.points['height']))

        return flippedImage, newBBoxes
    

class Flip(Filter):
    """
    Flips the image in X and Y axis
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        return cv2.flip(image, -1)

    def forwardWithBBox(self, image: ndarray, bBoxes: list[COCO]):
        flippedImage = self.forward(image)
        height, width = image.shape[:2]
        newBBoxes = []

        for bBox in bBoxes:
            newBBoxes.append(COCO(width-bBox.points['x']-bBox.points['width'], height-bBox.points['y']-bBox.points['height'], bBox.points['width'], bBox.points['height']))

        return flippedImage, newBBoxes