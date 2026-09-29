def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)
	"""
	# Your code here
	both = joint_distribution[('A', 'B')]
	not_a_b = joint_distribution[('`A', 'B')]
	p_b = both + not_a_b

	if p_b == 0:
		return 0.0
	return both / p_b