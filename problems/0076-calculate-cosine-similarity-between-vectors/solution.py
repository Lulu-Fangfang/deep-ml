import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	child = np.sum(v1 * v2)
	parent1 = np.sqrt(np.sum((np.abs(v1))**2))
	parent2 = np.sqrt(np.sum((np.abs(v2))**2))
	result = child / parent2 / parent1
	return float(result)