# from typing import Any, Union
# from Augmentors.SimpleAugmentor import SimpleAugmentor
# from Composite import Composite
# from Batch import Batch
# import os


# class MultiClassAugmentor(SimpleAugmentor):
#     """
#     Simple Augmentor:

#     - Scans the target folder
#     - Treats the folders inside as induvidual classes
#     - Gets all the images in the classes
#     - Augments them together
#     - Train, Valid, Test set partition if needed
#     """

#     def __init__(self, imagesDirectory: str, targetFolder: str, transforms: Composite, batchSize: int = 32, split: bool = True, ratio: tuple[float] = (0.75, 0.1, 0.15), seed: Union[None, Any] = None, imageDim: tuple[int, int] = (256, 256)) -> None:
#         """
#         Initializes Multi Class augmentor

#         Keyword arguments:

#         imagesDirectory (str) -- Path to the folder containing all the images

#         targetFolder (str) -- Path to the folder to store all the augmented images

#         transforms (Composite) -- Composition of filters

#         batchSize (int) -- Size of 1 batch. Each batch will be ran in a seperate batch. Choose your batch size wisely.

#         split (bool) -- Partitions the dataset of image into (Train, Test) or (Train, Test, Valid)

#         ratio (tuple[float]) -- Size of each partitions (in terms of batches) each value should be in range [0, 1] sum of all of the float should be 1

#         seed -- Seed for random generator

#         imageDim -- Dimension of the final image

#         Return: None
#         """
#         super().__init__(imagesDirectory, targetFolder,
#                          transforms, batchSize, split, ratio, seed, imageDim)
#         self.classes = [x for x in os.listdir(self.imagesDirectory) if os.path.isdir(os.path.join(self.imagesDirectory, x))]

#     @property
#     def targetImages(self) -> list[tuple[str, str]]:
#         """
#         List of all the images with their classes
#         """
#         images = []
#         for _cls in self.classes:
#             path = os.path.join(self.imagesDirectory, _cls)
#             subImgs = [(os.path.join(path, x), _cls) for x in os.listdir(path)]
#             images+= subImgs

#         return images
    
#     def partition(self) -> dict[str, list[tuple[str, str]]]:
#         return super().partition()
    
#     def batch(self, images: dict[str, list[str]] | list[str]):
#         def groupBatches(imgs:list[str], partition:str):
#             self.rand.shuffle(imgs)
#             for i in range(0, len(imgs), self.batchSize):
#                 bottom = i
#                 top = i + self.batchSize
#                 if top > len(imgs) - 1:
#                     top = len(imgs) - 1
#                 batch =  [x[0] for x in imgs[bottom:top]]
#                 print(batch)

#                 yield Batch(batch, os.path.join(self.targetFolder, partition), self.transforms, self.imageDim, f"{partition} #{i/self.batchSize}")

        
#         if isinstance(images, dict):
#             for partition in images:
#                 yield groupBatches(images[partition], partition)
#         else:
#             yield groupBatches(images, "")