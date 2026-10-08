import numpy as np

def rref(matrix):
	A = np.array(matrix, dtype=float, copy = True)
	tol = 1e-10
	if A.ndim != 2:
		return -1
	
	m , n = A.shape

	pivot_row = 0

	for col in range(n):

		if pivot_row == m :
			break
		best_row = pivot_row + np.argmax(np.abs(A[pivot_row:, col])
		)

		if abs(A[best_row, col]) <= tol:
			A[pivot_row, col] = 0.0
			continue

		A[[pivot_row, best_row], :] = A[[best_row, pivot_row], :]
		A[pivot_row , :] /= A[pivot_row, col]

		for row in range(m):
			if row != pivot_row:
				factor = A[row, col]
				A[row, :] -= factor * A[pivot_row, :]
				A[row, col] = 0.0

		pivot_row += 1
	
	A[np.abs(A) <= tol] = 0.0

	return A
