import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    m = len(batch_indices)

    predictions = X_batch @ weights + bias
    errors = predictions - y_batch

    grad_w = (2 / m) * (X_batch.T @ errors)
    grad_b = 2 * np.mean(errors)

    new_weights = weights - grad_w * lr
    new_b = bias - grad_b * lr

    return np.concatenate((new_weights, [new_b]))
    