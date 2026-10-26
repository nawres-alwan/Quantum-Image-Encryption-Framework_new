import numpy as np

class QIRBP:
    def __init__(self, image_array=None):
        """
        QIRBP representation for an image (grayscale or color).
        Stores pixel data (uint8 numpy array).
        """
        if image_array is not None:
            self.data = image_array.astype(np.uint8)
            self.shape = self.data.shape
        else:
            self.data = None
            self.shape = None
