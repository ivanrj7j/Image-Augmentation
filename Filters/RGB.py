from Filters.Filter import *

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
        shiftedImage  = np.clip(image + shift, 0, 255)
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