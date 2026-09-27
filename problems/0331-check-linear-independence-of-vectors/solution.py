import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    if not vectors:
        return True
   
   
    data = np.asarray(vectors)
    m, n = data.shape
    size = 0
    size = m

    rank = np.linalg.matrix_rank(data)
    if rank == size:
        return True
    else:
        return False 
    return True