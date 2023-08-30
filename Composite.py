from Filters import Filter
from typing import Union, Any, Callable
from random import Random
from numpy import ndarray
from COCO import COCO

class Composite:
    """
    Composes filters into one unit and randomly applies a filter when the transform() method is called
    """
    def __init__(self, filters:list[Filter], shouldApplyBBox=False, seed:Union[None, Any]=None, probablityFunction:Callable[[float], float] = lambda x: x, avoidPreviousFilter:bool = True) -> None:
        """
        Initializes the Composite Object
        
        Keyword arguments:
        filters (list[Filter]) -- A list of filters to apply
        shouldApplyBBox (bool) -- If the images have Bounding boxes
        seed (Any) -- Seed of random generator
        probablityFunction (Callable[[float], float]) -- Probablity distribution function of the graph, this could be any function f:[0, 1] -> [0, 1] **Important** The function should return values between 0 and 1, if the function returns more or less than that, the Composite **will not work**
        avoidPreviousFilter (bool) -- Avoids previous filter if set to true
        Return: None
        """
        
        self.filters = filters
        self.shouldApplyBBox = shouldApplyBBox

        if seed:
            for f in filters:
                f.setSeed(seed)
            self.rand = Random(seed)
        else:
            self.rand = Random()

        self.probablityFunction = probablityFunction

        if avoidPreviousFilter:
            self.previousIndex = -1

        self.avoidPreviousFilter = avoidPreviousFilter

    def pickIndex(self):
        x = self.rand.random()
        y = self.probablityFunction(x)
        if y>1 or y<0:
            raise ValueError("The probably function should return values from 0 to range ([0, 1])")
        z = y * len(self.filters)

        return min(list(range(len(self.filters))), key=lambda n: abs(n-z))
    
    def transform(self, image:ndarray, bBoxes:Union[list[COCO], None]=None):
        if self.avoidPreviousFilter:
            idx = self.previousIndex
            while idx == self.previousIndex:
                idx = self.pickIndex()
            self.previousIndex = idx
        else:
            idx = self.pickIndex()

        return self.filters[idx].apply(image, self.shouldApplyBBox, bBoxes)
        