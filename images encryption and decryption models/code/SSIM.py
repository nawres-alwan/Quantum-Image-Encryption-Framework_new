import numpy as np

def calculate_SSIM(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    This implementation computes a single SSIM value for the entire image (no sliding window), 
    optionally converting to grayscale for color images.
    SSIM formula (with constants C1, C2) for two signals x and y:
      SSIM = ((2μ_x μ_y + C1) * (2σ_{xy} + C2)) / ((μ_x^2 + μ_y^2 + C1) * (σ_x^2 + σ_y^2 + C2))
    """
    # If color, convert to grayscale for SSIM computation (luminance channel)
    if original.ndim == 3 and processed.ndim == 3:
        # Use a standard luminance conversion for fairness
        orig_gray = (0.2126*original[...,0] + 0.7152*original[...,1] + 0.0722*original[...,2]).astype(np.float64)
        proc_gray = (0.2126*processed[...,0] + 0.7152*processed[...,1] + 0.0722*processed[...,2]).astype(np.float64)
    else:
        orig_gray = original.astype(np.float64)
        proc_gray = processed.astype(np.float64)
    # Compute means
    mu_x = orig_gray.mean()
    mu_y = proc_gray.mean()
    # Compute variances and covariance
    sigma_x = orig_gray.var()
    sigma_y = proc_gray.var()
    sigma_xy = ((orig_gray - mu_x) * (proc_gray - mu_y)).mean()
    # SSIM constants (for 8-bit images, using K1=0.01, K2=0.03 as per standard)
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    # Compute SSIM
    ssim_num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_value = ssim_num / ssim_den
    return float(ssim_value)
