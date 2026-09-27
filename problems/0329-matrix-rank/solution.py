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
    for col in range(n):
        if rank == m:
            break
        
        pivot = rank + np.argmax(np.abs(B[rank: ,col]))
        if abs(B[pivot, col]) <= tol:
            continue
        
        B[[rank, pivot]] = B[[pivot, rank]]
        for row in range(rank + 1, m):
            factor = B[row, col] / B[rank, col]
            B[row, col:] -= factor * B[rank, col:]
        rank += 1

    return rank


    
