import numpy as np

class NEQR:
    def __init__(self, image_array=None):
        """
        NEQR representation for an image (originally grayscale, but we handle multi-channel too).
        Stores pixel data (uint8 numpy array).
        """
        if image_array is not None:
            self.data = image_array.astype(np.uint8)
            self.shape = self.data.shape
        else:
            self.data = None
            self.shape = None
