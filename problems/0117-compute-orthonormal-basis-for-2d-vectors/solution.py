import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    basis = []
    for v in vectors:
        u = np.array(v, dtype = float, copy = True)

        for q in basis:
            u = u - np.dot(q, u) * q
        
        norm = np.linalg.norm(u)

        if norm > tol:
            basis.append(u / norm)

        if len(basis) == 2:
            break
    return basis