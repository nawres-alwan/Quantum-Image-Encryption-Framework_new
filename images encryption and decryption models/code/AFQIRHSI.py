import numpy as np
import math

def rgb_to_hsi(image_rgb):
    """
    Convert an RGB image (uint8) to HSI components.
    Returns H, S, I as 2D numpy arrays (float64). H in [0,255] (0-360°), S in [0,255], I in [0,255].
    Uses standard RGB→HSI conversion formula:contentReference[oaicite:2]{index=2}.
    """
    image = image_rgb.astype(np.float64)
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    # Intensity: average of R, G, B
    I = (R + G + B) / 3.0
    # Saturation
    S = np.zeros_like(I)
    mask = I > 0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S[mask] = 1 - (min_rgb[mask] / I[mask])
    # Hue
    H = np.zeros_like(I)
    num = (R - 0.5 * G - 0.5 * B)
    den = np.sqrt((R - G) * (R - G) + (R - B) * (G - B))
    den_safe = den.copy()
    den_safe[den_safe == 0] = 1e-6  # avoid division by zero
    theta = np.degrees(np.arccos(np.clip(num / den_safe, -1, 1)))
    # If B > G, use 360° - theta
    H_val = theta.copy()
    H_val[B > G] = 360 - H_val[B > G]
    H[(I > 0) & (S > 0)] = H_val[(I > 0) & (S > 0)]
    # Scale H and S to 0-255
    H_scaled = (H / 360.0) * 255.0
    S_scaled = S * 255.0
    return H_scaled, S_scaled, I  # I is already 0-255 range

class AFQIRHSI:
    def __init__(self, image_array=None):
        """
        AFQIRHSI representation for a color image (RGB).
        Stores Hue, Saturation, Intensity components as uint8 arrays.
        """
        if image_array is not None:
            H_val, S_val, I_val = rgb_to_hsi(image_array)
            self.H = H_val.astype(np.uint8)
            self.S = S_val.astype(np.uint8)
            self.I = I_val.astype(np.uint8)
            self.shape = self.H.shape  # shape (H, W)
        else:
            self.H = None
            self.S = None
            self.I = None
            self.shape = None
