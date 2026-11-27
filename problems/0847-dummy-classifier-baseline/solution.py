import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    if strategy == "constant":
        return [constant] * n_test
    if strategy == "most_frequent":
        values, counts = np.unique(y_train, return_counts = True)
        result_num = values[np.argmax(counts)]
        return [result_num] * n_test
    
    if strategy == "uniform" :
        values, counts = np.unique(y_train, return_counts = True)
        result = [0] * n_test
        j = 0
        for i in range(n_test):
            if j >= len(values):
                j = j - len(values)
            result[i] = values[j]
            j += 1
        return result
    
    if strategy == "stratified":
        values, counts = np.unique(y_train, return_counts = True)
        counts_frac = counts / len(y_train)
        base_num = np.floor(counts_frac * n_test).astype(int)
        result = np.array([])
        result = np.repeat(values, base_num)
        
        remain_num = n_test - len(result)
        if remain_num == 0:
            return result
        frac, _ = np.modf(counts_frac * n_test)
        frac_cp = frac
        frac_cp = np.sort(frac_cp)[::-1]
        while len(result) < n_test:
            i = 0
            idx = np.flatnonzero(frac == frac_cp[i])

            result = np.insert(result, (base_num[idx] - 1), values[idx], axis = 0)
            i += 1
        return result



        


    
