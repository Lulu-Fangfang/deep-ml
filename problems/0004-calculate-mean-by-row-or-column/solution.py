import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	row_num = len(matrix)
	col_num = len(matrix[0])
	means = []
	if mode == 'column':
		result = [0] * col_num
		for i in range(row_num):
			for j in range(col_num):
				result[j] += matrix[i][j]
		
		for item in result:
			item_mean = item / row_num
			means.append(item_mean)
	elif mode == 'row':
		result = []
		for item in matrix:
			item_mean = np.sum(item)/ col_num
			result.append(item_mean)
		means = result
	

	
	return means