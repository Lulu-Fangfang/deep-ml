import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_aug = np.column_stack((X, np.ones(X.shape[0])))

    theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    W_new = theta[:-1].reshape(np.shape(W))
    b_new = np.asarray(theta[-1]).reshape(np.shape(b))

    if b_new.ndim == 0:
        b_new = b_new.item()
    return W_new, b_new
