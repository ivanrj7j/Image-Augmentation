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
        Applies Filter to the image, this method is meant to be a template for children to use
        
        Keyword arguments:
        image (ndarray) -- Numpy array of the image
        Return (ndarray) : Image with the filter applied
        """
        raise NotImplementedError("This method is meant to be implemented by the child")
    
    def forwardWithBBox(self, image:ndarray, bBoxes:list[COCO]):
        """
        Applies filter on the image, this method is meant to be a template for children to use
        
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
                image = np.mod(image, 255)
                return {'image': image, 'bBox': bBoxes}
            else:
                raise(TypeError("All elements of bBoxes should be COCO objects"))
        else:
            image = np.mod(self.forward(image), 255)
            return {'image':image}
        
    

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
    
    def forwardWithBBox(self, image: ndarray, bBoxes: list[COCO]):
        # anchor = np.multiply(image.shape[:2], self.rotateAnchor).astype(np.int64).tolist()
        # if self.maxAngle < 0:
        #     theta = self.rand.randint(0, self.maxAngle)
        # else:
        #     theta = self.rand.randint(0, self.maxAngle)

        # M = cv2.getRotationMatrix2D(anchor, theta, 1)

        # rotatedPoints = []
        # for bBox in bBoxes:
        #     corners = np.stack(tuple(map(lambda x: np.array(x), bBox.corners)))
        #     center = np.multiply(self.rotateAnchor, image.shape[:2])
            
        #     translatedCorners = corners - center

        #     rotationMatrix = np.array([
        #             [np.cos(theta), np.sin(theta)],
        #             [-np.sin(theta), np.cos(theta)]
        #         ])
            
        #     rotatedTranlatedCorners = np.dot(translatedCorners, rotationMatrix)

        #     newCorners = rotatedTranlatedCorners + center

        #     xMin = np.min(newCorners[:, 0])
        #     yMin = np.min(newCorners[:, 1])
        #     xMax = np.max(newCorners[:, 0])
        #     yMax = np.max(newCorners[:, 1])

        #     rotatedPoints.append(COCO.fromPascalVOCIterable((xMin, yMin, xMax, yMax)))


        # return cv2.warpAffine(image, M, image.shape[:2]), rotatedPoints

        pass


class RGBShift(Filter):
    """
    Shift the values of colors by a random amount
    """
    def __init__(self, rMax:int=25, gMax:int=25, bMax:int=25) -> None:
        super().__init__()        
        self.rMax = rMax
        self.gMax = gMax
        self.bMax = bMax

    def forward(self, image: ndarray) -> ndarray:
        rShift = self.rand.randint(-self.rMax, self.rMax)
        gShift = self.rand.randint(-self.gMax, self.gMax)
        bShift = self.rand.randint(-self.bMax, self.bMax)

        shift = np.array([rShift, gShift, bShift]).astype(np.uint8)
        shiftedImage  = image + shift
        return shiftedImage
    

class RGBPermute(Filter):
    """
    Permutes the RGB Channels
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: ndarray) -> ndarray:
        channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
        self.rand.shuffle(channels)
        return np.stack(channels, 2)
    

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