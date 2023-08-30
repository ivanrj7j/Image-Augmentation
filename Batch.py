from Composite import Composite
import os
from uuid import uuid1
from typing import Callable
import cv2
from numpy import ndarray

class Batch:
    """
    Augments a batch of Images, this does not support bounding boxes.

    Batches are meant to be ran in parallel in threads or async.
    """
    def __init__(self, targetImages:list[str], targetFolder:str, transforms:Composite, imageDim:tuple[int, int]=(256, 256)) -> None:
        """
        Initializes Batch Object
        
        Keyword arguments:
        targetImages (list[str]) -- List of paths to all of the images to be augmented
        targetFolder (str) -- The path to the folder to be saved
        transforms (Composite) -- A Composition of all the filters to be applied. Do not pass in transforms with bounding box
        Return: None
        """

        self.checkTransformCompatiblity(transforms)
        
        self.targetImages = targetImages
        self.targetFolder = targetFolder
        self.transforms = transforms
        self.imageDim = imageDim

    def checkTransformCompatiblity(self, transforms:Composite):
        if transforms.shouldApplyBBox:
            raise ValueError("This module does not support bounding boxes")

    def augment(self, variations:int=15, log:Callable[[str], None]=print):
        for image in self.targetImages:
            loadedImage = cv2.imread(image)
            if loadedImage is None:
                log(f"[NOT OPENABLE] {image} can't be loaded, image may be corrupted or the path is invalid")
                continue

            for _ in range(variations):
                try:
                    self.augmentImage(loadedImage)
                except Exception as e:
                    log(f"[AUGMENT ERROR] Cannot augment {image} due to [ [ {e} ] ]")

            try:
                self.originalImage(loadedImage)
            except Exception as e:
                log(f"[ORIGINAL IMAGE ERROR] Cannot save {image} due to [ [ {e} ] ]")

    def saveImage(self, image:ndarray, isOriginal:bool=False):
        """
        Saves the image to the destined path. This method is protected   
        
        Keyword arguments:
        image (ndarray) -- The ndarray of the image
        isOriginal (bool) -- If the image is the original image
        Return: None
        """
        
        name = f"original_{str(uuid1())}.jpg" if isOriginal else f"{str(uuid1())}.jpg"

        path = os.path.join(self.targetFolder, name)

        resizedImage = cv2.resize(image, self.imageDim)

        cv2.imwrite(path, resizedImage)


    def augmentImage(self, image:ndarray):
        """
        Augments the image
        
        Keyword arguments:
        image (ndarray) -- Ndarray of the image
        Return: None
        """
        transformed = self.transforms.transform(image)['image']
        self.saveImage(transformed)  

    def originalImage(self, image:ndarray):
        """
        Saves the original image

        Keyword arguments:
        image (ndarray) -- Ndarray of the image
        Return: None
        """
        self.saveImage(image, True)