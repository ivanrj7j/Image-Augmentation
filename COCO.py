from typing import Union

class COCO:
    def __init__(self, x:int, y:int, width:int, height:int) -> None:
        self.points = {'x':x, 'y':y, 'width':width, 'height':height}

    def normalize(self, shape:Union[tuple, list]):
        height_, width_ = shape[:2]
        x = self.points['x'] / width_
        y = self.points['y'] / height_
        width = self.points['width'] / width_
        height = self.points['height'] / height_

        return {'x':x, 'y':y, 'width':width, 'height':height}
        
