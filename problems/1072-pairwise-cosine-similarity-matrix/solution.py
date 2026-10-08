import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    x = np.asarray(X, dtype = float)

    norms = np.linalg.norm(X, axis = 1,keepdims = True)

    safe_norms = np.where(norms == 0, 1.0, norms)

    x_normalized = X / safe_norms

    S = x_normalized @ x_normalized.T

    np.fill_diagonal(S, (norms[:, 0] > 0).astype(float))

    return np.round(S, 4).tolist()