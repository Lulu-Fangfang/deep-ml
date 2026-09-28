import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	# Your code here
	dx = 0.0
	dy = 0.0
	dz = 0.0
	x = point[0]
	y = point[1]
	if len(point) == 3:
		z = point[2]
	match func_name:
		case 'poly2d':
			dx = x*2*y + y**2
			dy = x**2 + 2*x*y
		case 'exp_sum' :	
			dx = np.exp(x)
			dy = np.exp(y)
		case 'product_sin':
			dx = np.sin(y)
			dy = x * np.cos(y)
		case 'poly3d':
			dx = 2 *x *y
			dy = x **2 + z**2
			dz = 2 * y * z
		case 'squared_error':
			dx = 2 * (x - y)
			dy = (-2) * (x - y)

	if len(point) == 3:
		result = (dx,dy,dz)
	else:
		result = (dx,dy)
	return result