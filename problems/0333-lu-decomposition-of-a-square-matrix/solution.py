import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""

	A = np.asarray(A, dtype=float).copy()
	m,n = A.shape

	l = np.eye(n)
	u = A.copy()

	for k in range(n):
		if u[k, k] == 0:
			return  -1
		for i in range(k + 1, n):
			l[i, k] = u[i, k] / u[k,k]
			u[i, k:] -= l[i, k] * u[k, k:]
			u[i, k] = 0.0
	return (l , u) 
