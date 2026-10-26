import numpy as np
import math

class FRQI:
    def __init__(self, image_array=None):
        """
        FRQI representation for a grayscale image.
        If image_array is provided (2D numpy array), initialize theta angles.
        θ in [0, π/2] corresponds to pixel intensities [0, 255].
        """
        if image_array is not None:
            if image_array.ndim != 2:
                # If color, convert to grayscale by average
                image_array = np.mean(image_array, axis=2).astype(np.uint8)
            self.shape = image_array.shape
            # Map pixel value to angle: 0 → 0, 255 → π/2
            self.theta = (image_array.astype(np.float64) / 255.0) * (math.pi / 2)
        else:
            self.theta = None
            self.shape = None
