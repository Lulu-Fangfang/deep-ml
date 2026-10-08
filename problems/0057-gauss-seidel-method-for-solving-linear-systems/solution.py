import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	A = np.asarray(A, dtype=float)
	b = np.asarray(b, dtype=float).reshape(-1)

	m = A.shape[0]

	if x_ini is None:
		x = np.zeros(m, dtype = float)
	else:
		x = np.array(x_ini, dtype = float, copy = True).reshape(-1)

	for _ in range (n):
		for i in range(m):
			sum_before = np.dot(A[i, :i], x[:i])

			sum_after = np.dot(A[i, i + 1:], x[i + 1:])

			x[i] = (b[i] - sum_before - sum_after) / A[i, i]
	return x
