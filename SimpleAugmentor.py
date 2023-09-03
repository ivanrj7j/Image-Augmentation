class SimpleAugmentor:
    """
    Simple Augmentor:

    - Scans the target folder
    - Gets all the images in the folder
    - Augments them together
    - Train, Valid, Test set partition if needed
    - Does not support multiple classes
    """
    def __init__(self, targetFolder:str, batchSize:int=32, split:bool=True, ratio:tuple[float]=(0.75, 0.15, 0.1)) -> None:
        """
        Initializes simple augmentor
        
        Keyword arguments:

        targetFolder (str) -- Path to the folder containing all the images

        batchSize (int) -- Size of 1 batch. Each batch will be ran in a seperate batch. Choose your batch size wisely.

        split (bool) -- Partitions the dataset of image into (Train, Test) or (Train, Valid, Test)
        
        ratio (tuple[float]) -- Size of each partitions (in terms of batches) each value should be in range [0, 1] sum of all of the float should be 1
        
        Return: None
        """
        
        self.targetFolder= targetFolder
        self.batchSize = batchSize
        self.split = split

        if split:
            if not (isinstance(ratio, tuple) or isinstance(ratio, list)):
                raise TypeError("Ratio should be provided as a tuple or list")
            
            if len(ratio)<2 or len(ratio)>3:
                raise ValueError("There should either be 2 or 3 elements in ratio")
            
            if all(isinstance(x, float) or x==0 or x==1 for x in ratio):
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

    
        