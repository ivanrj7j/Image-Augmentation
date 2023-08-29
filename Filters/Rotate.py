from Filters.Filter import *

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