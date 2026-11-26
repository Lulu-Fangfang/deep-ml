import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:

    x_train = np.asarray(X_train, dtype = float)
    x_test = np.asarray(X_test, dtype = float)

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    std = np.where(std == 0, 1.0, std)

    return (x_test - mean) / std

