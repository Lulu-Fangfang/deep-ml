import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	if Y is None:
		X_centered = X - np.mean(X, axis=0)

		lengths = np.sqrt(np.sum(X_centered ** 2, axis = 0))
		X_normalized = X_centered / lengths

		result = X_normalized.T @ X_normalized

		return result
	else:
		x_centered = X - np.mean(X, axis=0)
		y_centered = Y - np.mean(Y, axis=0)
		x_lengths = np.sqrt(np.sum(x_centered ** 2, axis = 0))
		y_lengths = np.sqrt(np.sum(y_centered ** 2, axis = 0))
		x_normalized = x_centered / x_lengths
		y_normalized = y_centered / y_lengths

		result = x_normalized.T @ y_normalized
		return result