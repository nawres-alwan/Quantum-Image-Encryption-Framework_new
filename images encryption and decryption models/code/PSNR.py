import numpy as np

def calculate_PSNR(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) in decibels between two images.
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE), where MAX_I is max possible pixel value (255 for 8-bit images).
    Returns PSNR in dB. If images are identical, returns inf.
    """
    # Compute MSE first
    diff = original.astype(np.float64) - processed.astype(np.float64)
    mse_value = np.mean(diff**2)
    if mse_value == 0:
        return float('inf')  # no difference
    max_val = 255.0
    psnr_value = 20 * np.log10(max_val / np.sqrt(mse_value))
    return float(psnr_value)
