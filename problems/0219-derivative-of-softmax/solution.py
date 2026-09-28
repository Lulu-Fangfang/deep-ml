import numpy as np


def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	n = len(x)

	jacobian_matrix = np.zeros((n,n))

	softmax_sum = sum(np.exp(item) for item in x)
	s = [np.exp(item) / softmax_sum for item in x]

	for i in range(n):
		for j in range(n):
			if i == j :
				jacobian_matrix[i][j] = s[i] * (1 - s[i])
			else:
				jacobian_matrix[i][j] = -(s[i] * s[j])
	
	return jacobian_matrix

