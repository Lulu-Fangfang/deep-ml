import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    X = np.asarray(M, dtype=float).copy()

    norm = np.linalg.norm(X, ord = "fro")

    if norm == 0:
        return X.tolist()

    X /= norm

    for _ in range(num_iters):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)

    return X.tolist()
