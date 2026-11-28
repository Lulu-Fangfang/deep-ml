# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    y = np.asarray(y).reshape(-1,1)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    
    X = np.asarray(X, dtype = float)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    X_train = np.column_stack((np.ones(len(train_idx)),X_train))
    X_test = np.column_stack((np.ones(len(test_idx)), X_test))
    
    theta = np.linalg.lstsq(X_train, y[train_idx], rcond = None)[0]
    test_precisions = (X_test @ theta).ravel()
    
    return test_precisions, train_idx, val_idx, test_idx