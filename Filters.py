from numpy import ndarray
from random import Random
import cv2
import numpy as np
from typing import Union

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
    
    def forwardWithBBox(self, image:ndarray, bBox:tuple[int, int, int, int]):
        """
        Applies filter on the image, this method is meant to be a template for children to use
        
        Keyword arguments:
        image (ndarray) -- Numpy array of the image
        bBox (tuple) -- Tuple contating bounding box in COCO format [x, y, height, width]
        Return: Image with filter applied and and applied bbox
        """
        raise NotImplementedError("This method is meant to be implemented by the child")
    
    def apply(self, image:ndarray, shouldApplyBBox=False, bBox = Union[None, tuple]):
        if shouldApplyBBox and (isinstance(bBox, tuple) or isinstance(bBox, list)) and len(bBox) == 4:
            image, bBox = self.forwardWithBBox(image, bBox)
            return {'image': image, 'bBox': bBox}
        else:
            return {'image':self.forward(image)}
        
    

class Rotate(Filter):
    def __init__(self, maxAngle:int, rotateAnchor:tuple[float, float]) -> None:
        """
        Rotates the image in the given angle from the rotate angle
        
        Keyword arguments:
        maxAngle (int) -- Maximum angle of rotation
        rotateAnchor (tuple) -- Anchor of rotation values should be in range [0, 1] in form (x, y)
        Return: return_description
        """
        
        super().__init__()
        if not (isinstance(maxAngle, float) or isinstance(maxAngle, int)):
            raise TypeError("maxAngle should be float or int")
        self.maxAngle = round(maxAngle)

        if not (isinstance(rotateAnchor, tuple) or isinstance(rotateAnchor, list)):
            raise TypeError("rotateAnchor should be tuple or list")
        
        if len(rotateAnchor) != 2:
            raise ValueError(f"rotateAnchor should have 2 elements, x and y component not {len(rotateAnchor)}")
        
        for value, i in enumerate(rotateAnchor):
            if (not isinstance(value, float)) and (value != 0 and value != 1):
                raise TypeError("Rotate Anchor should either be float or 0 or 1")
            if value < 0 or value > 1:
                raise ValueError(f"The value of the {['x', 'y'][i]} element should be in range [0, 1]")
        
        self.rotateAnchor = rotateAnchor

    def forward(self, image: ndarray) -> ndarray:
        """
        Rotates the image by the given angle when initializing the object
        
        Keyword arguments:
        image -- The ndarray of the image to be rotated
        Return: The ndarray of the rotated image
        """
        anchor = np.multiply(image.shape[:2], self.rotateAnchor).astype(np.int64).tolist()
        if self.maxAngle < 0:
            M = cv2.getRotationMatrix2D(anchor, self.rand.randint(0, self.maxAngle), 1)
        else:
            M = cv2.getRotationMatrix2D(anchor, -self.rand.randint(0, self.maxAngle), 1)

        return cv2.warpAffine(image, M, image.shape[:2])

        
        