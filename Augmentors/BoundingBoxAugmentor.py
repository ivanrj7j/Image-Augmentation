import threading
import json
from Composite import Composite
from typing import Union, Any
from random import Random
from Batch import BoundingBoxBatch
import os
from uuid import uuid1


class BoundingBoxAugmentor:
    """
    Augments images in bounding box (COCO) format by taking a single json file as a parameter
    """

    def __init__(self, annotationsJsonPath: str, targetFolder: str, targetJsonPath: str, transforms: Composite, batchSize: int = 32, split: bool = True, ratio: tuple[float] = (0.75, 0.1, 0.15), seed: Union[None, Any] = None, imageDim: tuple[int, int] = (256, 256)) -> None:
        """
        Initializes the bounding box augmentor

        Keyword arguments:

        annotationsJsonPath (str) -- path to the json file contating augmentation data

        targetFolder (str) -- Path to the folder to store all the augmented images

        transforms (Composite) -- Composition of filters

        batchSize (int) -- Size of 1 batch. Each batch will be ran in a seperate batch. Choose your batch size wisely.

        split (bool) -- Partitions the dataset of image into (Train, Test) or (Train, Test, Valid)

        ratio (tuple[float]) -- Size of each partitions (in terms of batches) each value should be in range [0, 1] sum of all of the float should be 1

        seed -- Seed for random generator

        imageDim -- Dimension of the final image

        Return: None
        """

        self.annotationsJsonPath = annotationsJsonPath
        self.batchSize = batchSize
        self.targetFolder = targetFolder
        self.transforms = transforms
        self.split = split
        self.imageDim = imageDim
        self.targetJsonPath = targetJsonPath

        if split:
            if not (isinstance(ratio, tuple) or isinstance(ratio, list)):
                raise TypeError("Ratio should be provided as a tuple or list")

            if len(ratio) < 2 or len(ratio) > 3:
                raise ValueError(
                    "There should either be 2 or 3 elements in ratio")

            if not all(isinstance(x, float) or x == 0 or x == 1 for x in ratio):
                raise TypeError("The ratios should be float, 0 or 1")

            if sum(ratio) != 1:
                raise ValueError("The sum of ratio should be 1")

            if len(ratio) == 2:
                self.validate = False
            else:
                self.validate = True
                self.validRatio = ratio[2]

            self.trainRatio = ratio[0]
            self.testRatio = ratio[1]

        if seed is not None:
            self.rand = Random(seed)
        else:
            self.rand = Random()

    @property
    def data(self):
        with open(self.annotationsJsonPath, 'r') as f:
            return json.load(f)

    @property
    def targetImages(self):
        return [x['file_name'] for x in self.data['images']]

    def partition(self) -> dict[str, list[str]]:
        """
        Partitions the folders content into train, test, valid, if split set to true
        """
        images = self.targetImages

        self.rand.shuffle(images)
        # getting and shuffling all the images

        totalImages = len(images)

        trainSetLength = round(totalImages * self.trainRatio)
        trainSet = images[:trainSetLength]

        if self.validate:
            validSetLength = round(totalImages * self.validRatio)
            validInterval = validSetLength+trainSetLength
            validSet = images[trainSetLength:validInterval]

            testSet = images[validInterval:]

            return {"train": trainSet, "valid": validSet, "test": testSet}
        else:
            testSet = images[trainSetLength:]
            return {"train": trainSet, "test": testSet}

    def batch(self, images: Union[dict[str, list[str]], list[str]], multiThreaded: bool = False):
        """
        Groups the images into small batches with each batch of having size batchSize

        Keyword arguments:

        images (Union[dict[str, list[str]], list[str]]) -- All the path to images in list format, if the images are partitioned into groups, list of lists of path of images

        multiThreaded (bool) -- Takes measure to avoid file write issues when doing multi threading

        Return: List of batches
        """

        def groupBatches(imgs: list[str], partition: str):
            self.rand.shuffle(imgs)
            for i in range(0, len(imgs), self.batchSize):
                bottom = i
                top = i + self.batchSize
                if top > len(imgs) - 1:
                    top = len(imgs) - 1
                batch = imgs[bottom:top]

                if multiThreaded:
                    yield BoundingBoxBatch(batch, os.path.join(self.targetFolder, partition), self.annotationsJsonPath, os.path.join(os.path.dirname(self.targetJsonPath), f"temp-{str(uuid1())}.json"), self.transforms, self.imageDim, f"#{i/self.batchSize}")
                else:
                    yield BoundingBoxBatch(batch, os.path.join(self.targetFolder, partition), self.annotationsJsonPath, self.targetJsonPath, self.transforms, self.imageDim, f"#{i/self.batchSize}")

        if isinstance(images, dict):
            for partition in images:
                yield groupBatches(images[partition], partition)
        else:
            yield groupBatches(images, "")

    def createTargetFolder(self):
        """
        Creates target folder if it doesnt exist
        """

        if not os.path.exists(self.targetFolder):
            os.mkdir(self.targetFolder)

        if self.split:
            for p in ['train', 'test', 'valid']:
                path = os.path.join(self.targetFolder, p)
                if not os.path.exists(path):
                    os.mkdir(path)

    def sequentialAugment(self, variations: int = 15):
        """
        Augments every image one by one in 1 thread

        Keyword arguments:

        variations (int) -- Total Variations to the image

        Return: None        
        """

        self.createTargetFolder()

        if self.split:
            batches = self.batch(self.partition())
        else:
            batches = self.batch(self.targetImages)

        for _ in batches:
            for batch in _:
                batch.augment()

    def threadAugment(self, variations: int = 15):
        """
        Augments every image by dividing them into batches in parallel

        Keyword arguments:

        variations (int) -- Total Variations to the image

        Return: None        
        """

        self.createTargetFolder()

        if self.split:
            batches = self.batch(self.partition(), True)
        else:
            batches = self.batch(self.targetImages, True)

        threads: list[threading.Thread] = []

        for lst in batches:
            for batch in lst:
                thread = threading.Thread(
                    target=batch.augment, args=(variations,))
                threads.append(thread)

        for thread in threads:
            thread.start()

    def unifyTemps(self):
        """
        Unifies temporary json files. This method is meant to ran only after threadAugment is **finished running**
        """
        currentData = {
            "images": [],
            "categories": [],
            "annotations": []
        }
        basePath = os.path.dirname(self.targetJsonPath)
        for fileName in os.listdir(basePath):
            filePath = os.path.join(basePath, fileName)
            if os.path.isfile(filePath) and fileName.endswith(".json"):
                with open(os.path.join(filePath), 'r') as f:
                    data = json.load(f)
                    for d in currentData:
                        currentData[d] += data[d]
                os.remove(filePath)

        with open(self.targetJsonPath, 'w') as f:
            json.dump(currentData, f)
