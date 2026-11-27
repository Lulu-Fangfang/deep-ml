import numpy as np

def recall(y_true, y_pred):

    recall_v = 0
    y_co = y_true * y_pred
    v, c = np.unique(y_co, return_counts=True)
    if c.size > 1:
        tp = c[1]
    else:
        return 0.0
    value_true, count_true = np.unique(y_true, return_counts=True)
    if count_true[1]==0:
        recall_v = 0.0
    else:
        recall_v = tp / count_true[1]

    return recall_v
