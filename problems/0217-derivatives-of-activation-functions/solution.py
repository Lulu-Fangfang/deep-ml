import numpy as np


def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	sig_d = (1/ (1 + np.exp(-x))) - (1/ (1 + np.exp(-x)))**2

	tanh_d = -( (np.exp(x) - np.exp(-x))**2/ (np.exp(x) + np.exp(-x))**2 -1)

	relu_d = 1 if x > 0 else 0

	return {
		'sigmoid': sig_d,
		'tanh': tanh_d,
		'relu': relu_d
	}