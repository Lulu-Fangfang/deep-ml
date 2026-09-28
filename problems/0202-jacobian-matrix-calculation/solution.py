import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	if h < 0:
		return  -1
	
	n = len(x)
	m = len(f(x))

	jacobian = [[0.0] * n for _ in range(m)]

	for j in range(n):
		x_plus = x.copy()
		x_minus = x.copy()
		x_plus[j] += h
		x_minus[j] -= h

		y_plus = f(x_plus)
		y_minus = f(x_minus)

		for i in range(m):
			jacobian[i][j] = (y_plus[i] - y_minus[i]) / (2 * h)

	return jacobian