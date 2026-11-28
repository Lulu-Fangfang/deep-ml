import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    y_pred = []
    for sc in y_scores:
        if sc > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_true = np.asarray(y_true, dtype = int)
    y_pred = np.asarray(y_pred, dtype = int)
    final_re = y_pred * y_true
    v, c = np.unique(final_re, return_counts = True)
    v_t, c_t = np.unique(y_true, return_counts = True)
    v_p, c_p = np.unique(y_pred, return_counts = True)
    precision, recall = 0,0
    if c.size == 1 and v[0] == 0:
        return [0.0, 0.0]
    elif c.size == 1 and v[0] == 1:
        return [1.0, 1.0]

    if  c_p.size == 1:
        precision = 0.0
    else:
        precision= c[1] / c_p[1]
    if c_t.size == 1:
        recall = 0.0
    else:  
        recall = c[1] / c_t[1]

    return [precision, recall]
