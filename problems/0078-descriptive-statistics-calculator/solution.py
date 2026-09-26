import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    data_size = len(data)
    data_sum = np.sum(data)
    data_mean = data_sum / data_size

    data_sorted = data.sort()
    if data_size % 2 == 0:
        data_median = (data[data_size // 2] + data[(data_size // 2) - 1]) / 2
    else:
        data_median = (data[(data_size - 1) // 2])

    value, counts = np.unique(data, return_counts = True)
    most_common = value[np.argmax(counts)]
    varience_sum = 0
    for item in data:
        varience_item = (item - data_mean) ** 2
        varience_sum += varience_item
    varience_data = varience_sum / data_size

    standard_deviation = np.sqrt(varience_data)

    data_quene = np.percentile(data, [25, 50, 75])
    
    t25th_percentile = data_quene[0]
    t50th_percentile = data_quene[1]
    t75th_percentile = data_quene[2]
    
    interquartile_range = np.abs(t25th_percentile - t75th_percentile)

    return {
        'mean': data_mean,
        'median': data_median,
        'mode': most_common,
        'variance': varience_data,
        'standard_deviation': standard_deviation,
        '25th_percentile': t25th_percentile,
        '50th_percentile': t50th_percentile,
        '75th_percentile': t75th_percentile,
        'interquartile_range': interquartile_range
    }

    
    
