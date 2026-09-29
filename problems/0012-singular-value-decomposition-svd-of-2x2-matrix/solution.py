import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    A = np.asarray(A, dtype = float)

    B = A.T @ A
    a, b, d = B[0, 0], B[0, 1], B[1, 1]

    theta = 0.5 * np.arctan2(2 *b , a - d)
    c, s = np.cos(theta), np.sin(theta)

    V = np.array([[c, -s], [s, c]])

    AV = A @ V
    S = np.sqrt(np.sum(AV** 2, axis = 0))

    if S[0] > 0:
        u0 = AV[:, 0] / S[0]
        u1 = np.array([-u0[1], u0[0]])

        if np.dot(u1, AV[:,1]) < 0:
            u1 = -u1
    elif S[1]> 0:
        u1 = AV[:, 1] / S[1]
        u0 = np.array([u1[1], -u1[0]])
    else:
        u0 = np.array([1.0, 0.0])
        u1 = np.array([0.0, 1.0])
    U = np.column_stack((u0, u1))
    return U, S, V.T
