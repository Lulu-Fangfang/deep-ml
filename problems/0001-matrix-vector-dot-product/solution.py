import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	result = []
	if len(a[0]) != len(b):
		return -1
	for a0 in a:
		result0 = sum(a1 * b1 for a1,b1 in zip(a0,b))
		result.append(result0)
	return result