import numpy as np


def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	softmax_sum = sum(np.exp(item) for item in logits)
	softmax_list = [np.exp(i) / softmax_sum for i in logits]

	sm_array = np.asarray(softmax_list)
	y = np.zeros(len(logits))
	y[target] = 1

	gradient_array = sm_array - y

	return gradient_array
