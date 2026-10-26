import numpy as np

def calculate_ER(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Error Rate (ER) between two images.
    ER (also known as NPCR - Number of Pixels Change Rate) is the fraction of pixels that differ.
    """
    assert original.shape == processed.shape, "Images must have the same dimensions for ER."
    if original.ndim == 3:
        # For color images, consider a pixel "changed" if any channel changed
        diff_pixels = np.any(original != processed, axis=-1)
    else:
        diff_pixels = original != processed
    er = np.count_nonzero(diff_pixels) / diff_pixels.size
    return er
