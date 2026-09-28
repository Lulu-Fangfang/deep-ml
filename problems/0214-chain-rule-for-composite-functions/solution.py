import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	result = 1.0
	current = x
	for item in reversed(functions):
		match item:
			case 'sin':
				result *= np.cos(current)
				current = np.sin(current)
			case 'square':
				result *= 2 * current
				current = current **2
			case 'exp':
				result *= np.exp(current)
				current = np.exp(current)
			case 'log':
				if current < 0:
					return -1
				result *= 1 / current
				current = np.log(current)
			case _:
			
				return -1
	return result
	