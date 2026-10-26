import numpy as np

def calculate_HD(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Histogram Difference (HD) between two images.
    HD is computed as the sum of absolute differences between the histograms of the two images.
    For color images, the differences for each channel's histogram can be summed or averaged.
    Here, we average the HD across channels if color.
    """
    # Flatten channels and compute histograms
    if original.ndim == 3:
        # Color image: compute separate histograms for R, G, B
        total_diff = 0.0
        for ch in range(original.shape[2]):
            hist_orig, _ = np.histogram(original[..., ch].ravel(), bins=256, range=(0, 256))
            hist_proc, _ = np.histogram(processed[..., ch].ravel(), bins=256, range=(0, 256))
            total_diff += np.sum(np.abs(hist_orig - hist_proc))
        hd_value = total_diff / original.shape[2]
    else:
        hist_orig, _ = np.histogram(original.ravel(), bins=256, range=(0, 256))
        hist_proc, _ = np.histogram(processed.ravel(), bins=256, range=(0, 256))
        hd_value = np.sum(np.abs(hist_orig - hist_proc))
    return float(hd_value)
