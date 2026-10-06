import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a, b = matrix[0]
	c, d = matrix[1]

	trace = a + d
	determinant = a * d - b * c

	discriminant = trace ** 2 - 4 * determinant
	sqrt_discriminant = discriminant ** 0.5

	eigenvalue1 = (trace + sqrt_discriminant) / 2
	eigenvalue2 = (trace - sqrt_discriminant) / 2

	return [eigenvalue1, eigenvalue2]