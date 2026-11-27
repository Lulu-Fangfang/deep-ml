import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    result = np.array([])
    match strategy:
        case "mean":
            result = np.repeat(np.mean(y_train), n_test)
        case "median":
            result = np.repeat(np.median(y_train), n_test)
        case "quantile":
            if (quantile is None) or (quantile < 0) or (quantile > 1):
                raise ValueError(f"分位数非法：{quantile}")

            result = np.repeat(np.quantile(y_train, quantile), n_test)
        case "constant":
            if constant is None:
                raise ValueError(f"constant不合理：{constant}")
            result = np.repeat(constant, n_test)

        case _:
            raise ValueError(f"非正常策略")



    return result
            
