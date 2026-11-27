
import numpy as np

def rmse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"尺寸不匹配")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError(f"输入为空")
    rmse_res = np.sqrt(np.mean((y_pred - y_true)**2))


    return round(rmse_res,3)
