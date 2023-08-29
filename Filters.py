from numpy import ndarray
from random import Random
import cv2
import numpy as np

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
        

    def forward(self, image:ndarray) -> ndarray:
        """
        Applies Filter to the image, this method is meant to be a template for children to use
        
        Keyword arguments:
        image (ndarray) -- Numpy array of the image
        Return (ndarray) : Image with the filter applied
        """
        raise NotImplementedError("This method is meant to be implemented by the child")
    



        
        