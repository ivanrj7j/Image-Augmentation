from typing import Iterable

class COCO:
    """"
    Class for doing COCO bounding box calculations
    COCO uses the format [x, y, width, height] (Normally used for object detection)
    """
    def __init__(self, x:int, y:int, width:int, height:int) -> None:
        """
        Initializes the class with COCO format
        
        Keyword arguments:
        x (int) -- The x cordinate of the bounding box
        y (int) -- The y cordinate of the bounding box
        width (int) -- The width of the bounding box
        height (int) -- The height of the bounding box
        """
        
        self.points = {'x':round(x), 'y':round(y), 'width':round(width), 'height':round(height)}

    @classmethod
    def fromIterable(self, iterable:Iterable):
        """
        Initializes the class with iterable in COCO format
        
        Keyword arguments:
        iterable (Iterable) -- A iterable in the format [x, y, width, height]
        Return: self
        """

        if len(iterable) != 4:
            raise ValueError("The size of the iterable must be 4")
        
        return self(iterable[0], iterable[1], iterable[2], iterable[3])
    
    @classmethod
    def fromPascalVOCIterable(self, iterable:Iterable):
        """
        Initializes the class with iterable in Pascal VOC format
        
        Keyword arguments:
        iterable (Iterable) -- A iterable in the format [xMin, yMin, xMax, yMax]
        Return: self
        """

        if len(iterable) != 4:
            raise ValueError("The size of the iterable must be 4")
        
        return self(iterable[0], iterable[1], iterable[2]-iterable[0], iterable[3]-iterable[1])
    
    @classmethod
    def fromNormalizedIterable(self, iterable:Iterable, shape:Iterable):
        """
        Initializes the class with normalized iterable in COCO format
        
        Keyword arguments:
        iterable (Iterable) -- A iterable in the format [x, y, width, height]
        shape (Iterable) -- A iterable in the format [height, width]
        Return: self
        """

        height_, width_ = shape[:2]
        x = iterable[0] * width_
        y = iterable[1] * height_
        width = iterable[2] * width_
        height = iterable[3] * height_

        return self(x, y, width, height)
    

    @classmethod
    def fromNormalizedPascalVOCIterable(self, iterable:Iterable, shape:Iterable):
        """
        Initializes the class with normalized iterable in Pascal VOC format
        
        Keyword arguments:
        iterable (Iterable) -- A iterable in the format [xMin, yMin, xMax, yMax]
        shape (Iterable) -- A iterable in the format [height, width]
        Return: self
        """

        height_, width_ = shape[:2]
        x = iterable[0] * width_
        y = iterable[1] * height_
        width = (iterable[2] * width_) - x
        height = (iterable[3] * height_) - y

        return self(x, y, width, height)

        

    def normalize(self, shape:Iterable) -> dict[str, float]:
        """
        Normalizes the bounding box with respect to its image
        
        Keyword arguments:
        shape -- The shape of the image in tuple format (height, width)
        Return: Returns the points in normalized format in dictionary format
        """
        
        height_, width_ = shape[:2]
        x = self.points['x'] / width_
        y = self.points['y'] / height_
        width = self.points['width'] / width_
        height = self.points['height'] / height_

        return {'x':x, 'y':y, 'width':width, 'height':height}


    def toPascalVOC(self) -> dict[str, int]:
        """
        Converts the points to Pascal VOC ([xMin, yMin, xMax, yMax])
        """
        
        return {'xMin':self.points['x'], 'yMin':self.points['y'], 'xMax':self.points['x']+self.points['width'], 'yMax':self.points['y']+self.points['height']}
    
    def toNormalizedPascalVOC(self, shape:Iterable) -> dict[str, float]:
        """
        Normalizes the bounding box with respect to its image in Pascal VOC
        
        Keyword arguments:
        shape -- The shape of the image in tuple format (height, width)
        Return: Returns the points in normalized Pascal VOC format in dictionary format
        """

        height_, width_ = shape[:2]
        points = self.toPascalVOC()
        xMin = points['xMin'] / width_
        yMin = points['yMin'] / height_
        xMax = points['xMax'] / width_
        yMax = points['yMax'] / height_

        return {'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax}
    

    @property
    def iterableFormat(self):
        return self.points['x'], self.points['y'], self.points['width'], self.points['height']
    
    @property
    def iterablePascalVOCFormat(self):
        points = self.toPascalVOC()
        return points['xMin'], points['yMin'], points['xMax'], points['yMax']
    
    @property
    def corners(self):
        """
        Returns the corners of the rectangle in order (topLeft, bottomLeft, bottomRight, topRight)
        """
        
        return (self.points['x'], self.points['y']), (self.points['x'], self.points['y']+self.points['height']), (self.points['x']+self.points['width'], self.points['y']+self.points['height']), (self.points['x']+self.points['width'], self.points['y'])
        
