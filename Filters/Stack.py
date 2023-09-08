from COCO import COCO
from Filters import Filter
from numpy import ndarray

class Stack(Filter):
    """
    Stacks multiple filters together
    """
    def __init__(self, filters:list[Filter]) -> None:
        if not all(issubclass(type(x), Filter) for x in filters):
            raise TypeError("All the elements of the filters list should be a subclass of filters")
        
        self.filters = filters

    def forward(self, image: ndarray) -> ndarray:
        current = image
        for f in self.filters:
            current = f.forward(current)

        return current
    
    def forwardWithBBox(self, image: ndarray, bBoxes: list[COCO]):
        currentImage = image
        currentBBox = bBoxes

        for f in self.filters:
            currentImage, currentBBox = f.forwardWithBBox(currentImage, currentBBox)

        return currentImage, currentBBox       