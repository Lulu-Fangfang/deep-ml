import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	A = np.asarray(A, dtype = float)

	# if A.ndim != 2:
	# 	return -1

	m, n = A.shape
	# if m < n:
	# 	return -1
	
	Q = np.zeros((m, n))
	R = np.zeros((m, n))

	for j in range(n):

		v = A [:, j].copy()

		for i in range(j):
			R[i, j] = np.dot(Q[:, i], v)
			v = v - R[i, j ] * Q[:, i]

		R[j, j] = np.linalg.norm(v)

		if R[j, j] <= 1e-12 * np.linalg.norm(A[:, j]):
			return -1
		
		Q[:, j] = v / R[j, j]
	return Q, R