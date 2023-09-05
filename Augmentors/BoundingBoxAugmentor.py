import json

class BoundingBoxAugmentor:
    """
    Augments images in bounding box (COCO) format by taking a single json file as a parameter
    """
    def __init__(self, data:str) -> None:
        """
        Initializes the bounding box augmentor
        
        Keyword arguments:

        data (str) -- path to the json file contating augmentation data

        Return: return_description
        """
        
        with open(data, 'r') as f:
            self.data = json.load(f)

    
        