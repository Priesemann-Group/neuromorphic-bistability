import itertools
import numpy as np
import scipy.stats as stats 
from itertools import product


def shape_iter(shape):
    return product(*[range(n) for n in shape])

def ci(data, ci=0.95, p=0.5):
    data = data[~np.isnan(data)]
    data = np.sort(data)
    lowCount, upCount = stats.binom.interval(ci, data.size, p, loc=0)
    return data[int(lowCount) - 1], data[int(upCount)]

def median_ci(data):
    avg = np.zeros((3, data.shape[0]))
    avg[0] = np.nanmean(data, axis=1)
    # for i in range(data.shape[0]):
    #     avg[1:, i] = ci(data[i, :])
    return avg
