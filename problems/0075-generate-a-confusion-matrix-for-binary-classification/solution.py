
from collections import Counter

def confusion_matrix(data):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for _, value in enumerate(data):
        if value[0] == 1 and value[1] == 1:
            tp += 1
        if value[0] == 1 and value[1] == 0:
            fn += 1
        if value[0] == 0 and value[1] == 1:
            fp += 1
        if value[0] == 0 and value[1] == 0 :
            tn += 1
    return [[tp, fn],[fp, tn]]
