import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.asarray(A, dtype = float)
    b = np.asarray(b, dtype = float)

    det_A = np.linalg.det(A)

    if abs(det_A) < 1e-10:
        return -1

    n = A.shape[0]

    x = np.zeros(n, dtype=float)

    for i in range(n):
        A_i = A.copy()

        A_i[:, i] = b

        x[i] = np.linalg.det(A_i) / det_A

    return x