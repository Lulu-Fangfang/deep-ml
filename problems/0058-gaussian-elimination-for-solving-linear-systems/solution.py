import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	data = np.array(A, dtype = float , copy = True)
	b = np.asarray(b, dtype = float).reshape(-1,1)

	n = data.shape[0]
	data = np.append(data, b, axis = 1)

	for col in range(n):
		pivot = col + np.argmax(np.abs(data[col:, col]))

		if abs(data[pivot, col]) < 1e-12:
			return -1
		data[[col, pivot]] = data[[pivot, col]]

		for row in range(col + 1, n):
			factor = data[row, col] / data[col,col]
			data[row, col:] -= factor * data[col, col:]

	x = np.zeros(n)
	for row in range(n - 1, -1, -1):
		known_part = np.sum(data[row, row + 1:n] * x[row + 1:n])
		x[row] = (data[row, n]  - known_part) / data[row, row]
	return x
	


	
