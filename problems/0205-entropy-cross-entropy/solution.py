import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	P = np.asarray(P, dtype = float)
	Q = np.asarray(Q, dtype = float)


	mask = P > 0
	H_p = -np.sum(P[mask] * np.log(P[mask]))
	H_p_q = -np.sum(P[mask] * np.log(Q[mask]))
	return float(H_p), float(H_p_q)