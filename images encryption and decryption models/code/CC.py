import numpy as np

def calculate_CC(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Correlation Coefficient (CC) between two images.
    Computes the Pearson correlation coefficient between flattened pixel arrays of the two images.
    A value near 0 indicates no correlation (good for encryption), while 1 or -1 indicates high correlation.
    """
    x = original.astype(np.float64).ravel()
    y = processed.astype(np.float64).ravel()
    # Compute Pearson correlation: cov(x,y) / (std(x)*std(y))
    x_mean = x.mean()
    y_mean = y.mean()
    # If either image is constant, correlation is undefined (return 0 in that case)
    x_std = x.std()
    y_std = y.std()
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    cc_value = cov_xy / (x_std * y_std)
    return float(cc_value)
