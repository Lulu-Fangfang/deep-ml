import numpy as np

def to_categorical(x, n_col=None):
    col_num = 0
    row_num = len(x)

    if n_col == None:
        col_num = (x[np.argmax(x)] + 1)
    else:
        col_num = n_col

    result = np.zeros((row_num, col_num))
    for idx , value in enumerate(x):
        result[idx][value] = 1

    return result
