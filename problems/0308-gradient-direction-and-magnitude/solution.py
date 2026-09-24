import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here

	gradient_array = np.asarray(gradient , dtype = float)
	magnitude = np.linalg.norm(gradient_array, ord=2)

	if magnitude == 0:
		direction = np.zeros_like(gradient_array)
		descent_direction = np.zeros_like(gradient_array)
	else:
		direction = gradient_array / magnitude
		descent_direction = - direction

	return {
		"magnitude": float(magnitude),
		"direction": direction.tolist(),
		"descent_direction": descent_direction.tolist(),
	}
