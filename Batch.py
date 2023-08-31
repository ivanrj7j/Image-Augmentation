from Composite import Composite
import os
from uuid import uuid1
from typing import Callable
import cv2
from numpy import ndarray
import json
from COCO import COCO
import numpy as np
from typing import Union


class Batch:
    """
    Augments a batch of Images, this does not support bounding boxes.

    Batches are meant to be ran in parallel in threads or async.
    """

    def __init__(self, targetImages: list[str], targetFolder: str, transforms: Composite, imageDim: tuple[int, int] = (256, 256)) -> None:
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

    def checkTransformCompatiblity(self, transforms: Composite):
        if transforms.shouldApplyBBox:
            raise ValueError("This module does not support bounding boxes")

    def augment(self, variations: int = 15, log: Callable[[str], None] = print):
        """
        Augments the batch

        Keyword arguments:
        variations (int) -- The Number of variations of images
        log (Callable) -- A Function which takes in a string, this is used to log errors 
        Return: None
        """

        for image in self.targetImages:
            loadedImage = cv2.imread(image)
            # loading image

            if loadedImage is None:
                log(f"[NOT OPENABLE] {image} can't be loaded, image may be corrupted or the path is invalid")
                continue
                # error logging if image doesnt exist

            for _ in range(variations):
                try:
                    self.augmentImage(loadedImage)
                except Exception as e:
                    log(
                        f"[AUGMENT ERROR] Cannot augment {image} due to [ [ {e} ] ]")
                # trying to annotate image

            try:
                self.originalImage(loadedImage)
            except Exception as e:
                log(
                    f"[ORIGINAL IMAGE ERROR] Cannot save {image} due to [ [ {e} ] ]")
            # trying to save original image

    def saveImage(self, image: ndarray, isOriginal: bool = False):
        """
        Saves the image to the destined path. This method is protected   

        Keyword arguments:
        image (ndarray) -- The ndarray of the image
        isOriginal (bool) -- If the image is the original image
        Return: None
        """

        name = f"original_{str(uuid1())}.jpg" if isOriginal else f"{str(uuid1())}.jpg"
        path = os.path.join(self.targetFolder, name)
        # getting name and path

        resizedImage = cv2.resize(image, self.imageDim)
        # resizing image

        cv2.imwrite(path, resizedImage)
        # saving image

    def augmentImage(self, image: ndarray):
        """
        Augments the image

        Keyword arguments:
        image (ndarray) -- Ndarray of the image
        Return: None
        """
        transformed = self.transforms.transform(image)['image']
        self.saveImage(transformed)
        # transforming and saving image

    def originalImage(self, image: ndarray):
        """
        Saves the original image

        Keyword arguments:
        image (ndarray) -- Ndarray of the image
        Return: None
        """
        self.saveImage(image, True)
        # just saving image with original set to true


class BoundingBoxBatch(Batch):
    def __init__(self, targetImages: list[str], targetFolder: str, annotationsJson: str, targetJsonPath: str, transforms: Composite, imageDim: tuple[int, int] = (256, 256)) -> None:
        """
        Initializes Batch Object

        Keyword arguments:
        targetImages (list[str]) -- List of paths to all of the images to be augmented
        targetFolder (str) -- The path to the folder to be saved
        annotationsJson (str) -- The path to the json file to where all the annotations exists
        targetJsonPath (str) -- The path to json file to save augmented annotations
        transforms (Composite) -- A Composition of all the filters to be applied. Do not pass in transforms with bounding box
        imageDim (tuple[width, height]) -- A tuple containing width and height of the image to be resized to
        Return: None
        """
        super().__init__(targetImages, targetFolder, transforms, imageDim)
        with open(annotationsJson, 'r') as f:
            self.annotations = json.load(f)
        self.targetJsonPath = targetJsonPath

    def checkTransformCompatiblity(self, transforms: Composite):
        if not transforms.shouldApplyBBox:
            raise ValueError("The transforms should have bounding box enabled")

    def getOGID(self, imageName: str) -> list[COCO]:
        fileName = os.path.split(imageName)[-1]
        # getting filename

        imgData = list(
            filter(lambda x: fileName in x['file_name'], self.annotations['images']))[0]

        return imgData['id']
        # getting the id

    def getBBoxes(self, imageName:str) -> list[COCO, Union[int, str]]:
        imgID = self.getOGID(imageName)

        imgAnnotation = list(
            filter(lambda x: x['image_id'] == imgID, self.annotations['annotations']))

        annotations = [(COCO.fromIterable(x['bbox']), x['category_id']) for x in imgAnnotation]

        return annotations

    def augment(self, variations: int = 15, log: Callable[[str], None] = print):
        """
        Augments the batch

        Keyword arguments:
        variations (int) -- The Number of variations of images
        log (Callable) -- A Function which takes in a string, this is used to log errors 
        Return: None
        """

        for image in self.targetImages:
            loadedImage = cv2.imread(image)
            # loading image

            if loadedImage is None:
                log(f"[NOT OPENABLE] {image} can't be loaded, image may be corrupted or the path is invalid")
                continue
                # error logging if image doesnt exist

            bBoxes = self.getBBoxes(image)

            for _ in range(variations):
                try:
                    self.augmentImage(loadedImage, bBoxes)
                except Exception as e:
                    log(
                        f"[AUGMENT ERROR] Cannot augment {image} due to [ [ {e} ] ]")
                # trying to annotate image

            try:
                self.originalImage(loadedImage, bBoxes)
            except Exception as e:
                log(
                    f"[ORIGINAL IMAGE ERROR] Cannot save {image} due to [ [ {e} ] ]")
            # trying to save original image

    def augmentImage(self, image: ndarray, bBoxes: list[COCO]):
        """
        Augments the image

        Keyword arguments:
        image (ndarray) -- Ndarray of the image
        bBoxes (list[COCO]) -- List of all the bounding boxes
        Return: None
        """
        transformed = self.transforms.transform(image, [x[0] for x in bBoxes])
        self.saveImage(transformed['image'], list(zip(transformed['bBox'], [x[1] for x in bBoxes])))
        # transforming and saving image

    def originalImage(self, image: ndarray, bBoxes: list[COCO]):
        """
        Saves the original image

        Keyword arguments:
        image (ndarray) -- Ndarray of the image
        bBoxes (list[COCO]) -- List of all the bounding boxes
        Return: None
        """
        self.saveImage(image, bBoxes, True)
        # just saving image with original set to true

    def saveImage(self, image: ndarray, bBoxes: list[COCO], isOriginal: bool = False):
        """
        Saves the image to the destined path. This method is protected   

        Keyword arguments:
        image (ndarray) -- The ndarray of the image
        bBoxes (list[COCO]) -- List of all the bounding boxes
        isOriginal (bool) -- If the image is the original image
        Return: None
        """

        id_ = f"original_{str(uuid1())}" if isOriginal else str(uuid1())
        path = os.path.join(self.targetFolder, id_+'.jpg')
        # getting id and path

        ogHeight, ogWidth = image.shape[:2]
        # getting the original height and width

        og = np.array((ogWidth, ogHeight))
        new = np.array(self.imageDim)

        ratio = new/og

        newBBoxes:list[tuple[COCO, Union[int, str]]] = []

        for bBox, categoryID in bBoxes:
            x1, y1, x2, y2 = bBox.iterablePascalVOCFormat
            points = np.array([
                [x1, y1],
                [x2, y2]
            ])

            newPoints = points * ratio

            newBBoxes.append((COCO.fromPascalVOCIterable(newPoints.flatten().astype(np.int64).tolist()), categoryID))


        resizedImage = cv2.resize(image, self.imageDim)
        # resizing image

        cv2.imwrite(path, resizedImage)
        # saving image

        if os.path.isfile(self.targetJsonPath):
            with open(self.targetJsonPath, 'r') as f:
                currentData = json.load(f)
        else:
            currentData = {
                "images": [],
                "categories": self.annotations['categories'],
                "annotations": []
            }

        currentData['images'].append(
            {
                "width": self.imageDim[0],
                "height": self.imageDim[1],
                "id": id_,
                "file_name": path
            }
        )

        for bBox, categoryID in newBBoxes:
            currentData['annotations'].append(
                {
                    "id": str(uuid1()),
                    "image_id": id_,
                    "category_id": categoryID,
                    "segmentation": [],
                    "bbox": bBox.iterableFormat,
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": bBox.points['width'] * bBox.points['height']
                }
            )

        with open(self.targetJsonPath, 'w') as f:
            json.dump(currentData, f)