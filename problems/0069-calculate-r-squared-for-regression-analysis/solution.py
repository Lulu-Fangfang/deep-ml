
import numpy as np

def r_squared(y_true, y_pred):
    result = 1 - ((np.sum((y_true - y_pred) ** 2))/np.sum((y_true - np.mean(y_true)) ** 2))
    return result
