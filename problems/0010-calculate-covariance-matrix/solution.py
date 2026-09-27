import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

	n = len(vectors)
	m = len(vectors[0])
	data_np = np.empty((0, m), dtype = float)
	for item in range(n):
		data_sum = np.sum(vectors[item])
		mean = data_sum / len(vectors[item])
		
		centered = np.asarray(vectors[item], dtype=float) - mean

		data_np = np.vstack((data_np, centered))
	
	result = np.zeros((n,n), dtype = float)

	for i in range(n):
		for j in range(n):
			result[i,j] = np.sum(data_np[i] * data_np[j] / (m - 1))
	return result
	
	