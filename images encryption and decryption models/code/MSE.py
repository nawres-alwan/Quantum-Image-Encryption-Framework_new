import numpy as np

def calculate_MSE(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two images.
    MSE = average of (original - processed)^2 over all pixels and channels.
    """
    diff = original.astype(np.float64) - processed.astype(np.float64)
    mse_value = np.mean(diff**2)
    return float(mse_value)
