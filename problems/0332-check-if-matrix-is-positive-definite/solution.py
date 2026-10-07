import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    positive_flag = False
    if np.linalg.det(matrix) > 0:
        positive_flag = True
    eigenvalues = np.linalg.eigvals(matrix)
    if np.allclose(eigenvalues.imag, 0):
        eigenvalues = sorted(
            round(float(x), 4) for x in eigenvalues.real
        )
    # eigenvalues.sort()
    return {
        'is_positive_definite': positive_flag,
        'eigenvalues': eigenvalues
    }