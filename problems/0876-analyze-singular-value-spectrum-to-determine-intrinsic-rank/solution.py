import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	s = np.linalg.svd(delta_W, compute_uv = False)

	if s.size == 0 or s[0] == 0:
		return 0
	
	energy = (s / s[0]) **2
	cumulative_energy = np.cumsum(energy)

	target = energy_threshold * cumulative_energy[-1]

	index = np.searchsorted(cumulative_energy, target, side="left")

	return int(index + 1)
