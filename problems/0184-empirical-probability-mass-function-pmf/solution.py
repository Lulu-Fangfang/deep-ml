import numpy as np



def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    if not isinstance(samples, np.ndarray):
        samples = np.asarray(samples)

    values, count = np.unique(samples, return_counts = True)
    samples_size = len(samples)
    percent = []
    for item in count:
        item_percent = item / samples_size
        percent.append(item_percent)
    return list(zip(values.tolist(), percent))