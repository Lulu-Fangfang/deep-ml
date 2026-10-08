import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    data = np.asarray(observations)
    successes = np.sum(data)
    failures = len(data) - successes

    alpha_post = alpha + successes
    beta_post = beta + failures

    if alpha_post == 1 and beta_post == 1:
        return 0.5

    estimate = (alpha_post -1 ) / (alpha_post + beta_post - 2)

    return round(float(estimate), 4)