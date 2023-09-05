from random import Random
from typing import Any, Union
import os
from Batch import Batch
from Composite import Composite

class SimpleAugmentor:
    """
    Simple Augmentor:

    - Scans the target folder
    - Gets all the images in the folder
    - Augments them together
    - Train, Valid, Test set partition if needed
    - Does not support multiple classes
    """
    def __init__(self, imagesDirectory:str, targetFolder:str, transforms:Composite, batchSize:int=32, split:bool=True, ratio:tuple[float]=(0.75, 0.1, 0.15), seed:Union[None, Any]=None, imageDim:tuple[int, int]=(256, 256)) -> None:
        """
        Initializes simple augmentor
        
        Keyword arguments:

        imagesDirectory (str) -- Path to the folder containing all the images
        
        targetFolder (str) -- Path to the folder to store all the augmented images

        transforms (Composite) -- Composition of filters

        batchSize (int) -- Size of 1 batch. Each batch will be ran in a seperate batch. Choose your batch size wisely.

        split (bool) -- Partitions the dataset of image into (Train, Test) or (Train, Test, Valid)
        
        ratio (tuple[float]) -- Size of each partitions (in terms of batches) each value should be in range [0, 1] sum of all of the float should be 1

        seed -- Seed for random generator

        imageDim -- Dimension of the final image
        
        Return: None
        """
        
        self.imagesDirectory= imagesDirectory
        self.batchSize = batchSize
        self.targetFolder = targetFolder
        self.transforms = transforms
        self.split = split
        self.imageDim = imageDim

        if split:
            if not (isinstance(ratio, tuple) or isinstance(ratio, list)):
                raise TypeError("Ratio should be provided as a tuple or list")
            
            if len(ratio)<2 or len(ratio)>3:
                raise ValueError("There should either be 2 or 3 elements in ratio")
            
            if not all(isinstance(x, float) or x==0 or x==1 for x in ratio):
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
    def targetImages(self) -> list[str]:
        """
        List of all the images that will be target.
        """
        return os.listdir(self.imagesDirectory)
    
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

            return {"train": trainSet, "valid":validSet, "test":testSet}
        else:
            testSet = images[trainSetLength:]
            return {"train":trainSet, "test":testSet}
        
    def batch(self, images:Union[dict[str, list[str]], list[str]]):
        """
        Groups the images into small batches with each batch of having size batchSize
        
        Keyword arguments:

        images (Union[dict[str, list[str]], list[str]]) -- All the path to images in list format, if the images are partitioned into groups, list of lists of path of images

        Return: List of batches
        """

        def groupBatches(imgs:list[str], partition:str):
            self.rand.shuffle(imgs)
            for i in range(0, len(imgs), self.batchSize):
                bottom = i
                top = i + self.batchSize
                if top > len(imgs) - 1:
                    top = len(imgs) - 1
                batch =  list(map(lambda x: os.path.join(self.imagesDirectory, x), imgs[bottom:top]))

                yield Batch(batch, os.path.join(self.targetFolder, partition), self.transforms, self.imageDim, f"Batch {i/self.batchSize}")

        
        if isinstance(images, dict):
            for partition in images:
                yield groupBatches(images[partition], partition)
        else:
            yield groupBatches(images, "")

    def sequentialAugment(self, variations:int=15):
        """
        Augments every image one by one in 1 thread
        
        Keyword arguments:

        variations (int) -- Total Variations to the image

        Return: None        
        """

        if not os.path.exists(self.targetFolder):
            os.mkdir(self.targetFolder)

        batches:list[Batch] = []


        if self.split:
            partitions = self.partition()
            for partition in partitions:

                partitionPath = os.path.join(self.targetFolder, partition)
                if not os.path.exists(partition):
                    os.mkdir(partitionPath)
                
                images = list(map(lambda x: os.path.join(self.imagesDirectory, x), partitions[partition]))
                batch = Batch(images, partitionPath, self.transforms, self.imageDim, f"{partition}")
                batches.append(batch)
        else:
            images = list(map(lambda x: os.path.join(self.imagesDirectory, x), self.targetImages))
            batch = Batch(images, self.targetFolder, self.transforms, self.imageDim)

            batches.append(batch)

        for batch in batches:
            batch.augment(variations)