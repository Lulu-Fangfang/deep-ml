import numpy as np

def min_max(x: list[float]) -> list[float]:
    idx_max = np.argmax(x)
    idx_min = np.argmin(x)
    x_max = x[idx_max]
    x_min = x[idx_min]
    length = len(x)
    if x_max == x_min:
        return [0.0] * length

    for idx, _ in enumerate(x):
        x[idx] = (x[idx] - x_min)/ (x_max - x_min)

    return x
