import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: Weight of recall relative to precision (beta > 1 favours recall)
    :return: F-Score rounded to three decimal places
    """
    true_target = y_true * y_pred
    v, c = np.unique(true_target, return_counts = True)
    if c.size == 1:
        return 0.0
    t_v, t_c = np.unique(y_true, return_counts = True)
    p_v, p_c = np.unique(y_pred, return_counts = True)
    recall_v = c[1] / t_c[1]
    pre_v = c[1] / p_c[1]

    F_sc = (1 + beta **2) * (pre_v * recall_v) / (beta **2 * pre_v + recall_v)

    return round(F_sc, 3)
