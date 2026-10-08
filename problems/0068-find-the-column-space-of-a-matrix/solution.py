
import numpy as np

def matrix_image(A):
	# Write your code here

	A = np.asarray(A, dtype=float)

	rank = np.linalg.matrix_rank(A)

	basis = A[:, :0]


	for i in range(A.shape[1]):
		if basis.shape[1] == rank:
			break
		candidate = np.column_stack((basis, A[:, i]))

		if np.linalg.matrix_rank(candidate) > basis.shape[1]:
			basis = candidate

	return basis