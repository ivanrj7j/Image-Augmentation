from numpy import ndarray
from random import Random
import cv2
import numpy as np
from typing import Union, Any
from COCO import COCO
import math

class Filter:
    def __init__(self) -> None:
        """
        Base Class for all the filters
        """
        
        self.rand = Random()

    def setSeed(self, seed) -> None:
        """
        Sets a seed for the random generator 
        
        Keyword arguments:

        seed (any) -- The seed for the random generator
        """
        self.rand = Random(seed)
        self.seed = seed
        

    def forward(self, image:ndarray) -> ndarray:
        """
        Applies Filter to the image
        
        Keyword arguments:

        image (ndarray) -- Numpy array of the image

        Return (ndarray) : Image with the filter applied
        """
        raise NotImplementedError("This method is meant to be implemented by the child")
    
    def forwardWithBBox(self, image:ndarray, bBoxes:list[COCO]):
        """
        Applies filter on the image
        
        Keyword arguments:

        image (ndarray) -- Numpy array of the image

        bBoxes (list[COCO]) -- List Containg bounding boxes in COCO format

        Return: Image with filter applied and and applied bbox
        """
        return self.forward(image), bBoxes
    
    def apply(self, image:ndarray, shouldApplyBBox=False, bBoxes:Union[None, list[COCO]]=None) -> dict[str, Any]:
        """
        Applies filter to the image and returns a dictionary with all the data
        
        Keyword arguments:

        image (ndarray) -- Ndarray of the image

        shouldApplyBBox (bool) -- If the bounding boxes should be applied

        bBoxes (list[COCO]) -- List Containg bounding boxes in COCO format

        Return: dictionary of the values to be returned
        """
        
        if shouldApplyBBox:
            if all(isinstance(x, COCO) for x in bBoxes):
                image, bBoxes = self.forwardWithBBox(image, bBoxes)
                return {'image': image, 'bBox': bBoxes}
            else:
                raise(TypeError("All elements of bBoxes should be COCO objects"))
        else:
            image = self.forward(image)
            return {'image':image}