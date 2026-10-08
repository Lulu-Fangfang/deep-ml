import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    mu_p = np.asarray(mu_p, dtype = float)
    mu_q = np.asarray(mu_q, dtype = float)

    cov_p = np.asarray(Cov_p, dtype = float)
    cov_q = np.asarray(Cov_q, dtype = float)

    d = mu_p.size

    for array in (mu_p, mu_q, cov_p, cov_q):
        if not np.all(np.isfinite(array)):
            return -1
    
    for cov in (cov_p, cov_q):
        if not np.allclose(cov, cov.T):
            return -1
        try: 
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            return -1
    

    _, logdet_p = np.linalg.slogdet(cov_p)
    _, logdet_q = np.linalg.slogdet(cov_q)

    logdet_term = logdet_q - logdet_p

    trace_term = np.trace(np.linalg.solve(cov_q, cov_p))

    diff = mu_q - mu_p
    mean_term = diff @ np.linalg.solve(cov_q, diff)

    kl = 0.5 * (logdet_term - d + trace_term + mean_term)
    return float(kl)