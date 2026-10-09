import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:

	B_matrix = np.array(B, dtype=float)
	C_matrix = np.array(C, dtype=float)
	return np.linalg.solve(C_matrix, B_matrix)