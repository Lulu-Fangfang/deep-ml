def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	if len(priors) == 2:
		postpriors = (priors[0] * likelihoods[0])/ (priors[0] * likelihoods[0] + priors[1] * likelihoods[1])
	elif len(priors) == 3:
		postpriors = (priors[0] * likelihoods[0])/ (priors[0] * likelihoods[0] + priors[1] * likelihoods[1] + priors[1] * likelihoods[1])
		return postpriors, ((1 - postpriors)/2),((1 - postpriors)/2) 
	return postpriors, (1 - postpriors)
