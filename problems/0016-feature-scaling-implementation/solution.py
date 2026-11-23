import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    x = np.asarray(data)

    mean = x.mean(axis = 0)
    std = x.std(axis = 0, ddof = 0)

    col_min = x.min(axis = 0)
    col_range = x.max(axis = 0) - col_min

    standardized = (x - mean) / np.where(std == 0, 1.0, std)

    normalized = (x - col_min) / np.where(col_range  == 0, 1.0, col_range)

    return np.round(standardized, 4), np.round(normalized, 4)