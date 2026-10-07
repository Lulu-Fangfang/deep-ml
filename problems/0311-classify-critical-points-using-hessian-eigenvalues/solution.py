import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    value = np.linalg.eigvals(hessian)
    all_positive = np.all(value > 0)
    if all_positive:
        return -1
    
    all_negitive = np.all(value < 0)
    if all_negitive:
        return 1

    if np.any(np.abs(value) <= tol):
        return None
    if np.any(value > tol) and np.any(value < -tol):
        return 0
