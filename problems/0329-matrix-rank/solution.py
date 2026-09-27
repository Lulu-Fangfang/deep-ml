import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    B = np.array(A, dtype=float, copy = True)
    m, n = B.shape
    rank = 0
    
    return np.linalg.matrix_rank(A)
