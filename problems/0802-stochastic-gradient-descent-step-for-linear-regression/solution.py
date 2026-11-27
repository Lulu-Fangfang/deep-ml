import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    w = weights.astype(float)
    n_sample = len(X)

    for i in range(n_iter):
        j = i % n_sample
        x = X[j]
        prediction = x @ w
        gradiant = 2 * (prediction - y[j]) * x
        w -= learning_rate * gradiant

    return w.tolist()