import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Your code here
    result = np.array([])

    z_v = X @ weights + bias
    sigmoid = lambda z_v: 1 / (1 + np.exp(-z_v))
    p_v = sigmoid(z_v)
    for idx, value in enumerate(p_v):
        if value >= 0.5:
            result = np.append(result, 1)
        if value < 0.5:
            result = np.append(result, 0)
    return result