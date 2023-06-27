import itertools
import numpy as np
import scipy.stats as stats 
from itertools import product
from sklearn.metrics import mutual_info_score

from scipy.stats import sem, t

def mutual_info(x, y, bins=np.arange(3)):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

def shape_iter(shape):
    return product(*[range(n) for n in shape])

def ci(data, ci=0.95, p=0.5):
    data = data[~np.isnan(data)]
    data = np.sort(data)
    lowCount, upCount = stats.binom.interval(ci, data.size, p, loc=0)
    return data[int(lowCount) - 1], data[int(upCount)]

def median_ci(data):
    avg = np.zeros((3, data.shape[0]))
    avg[0] = np.nanmedian(data, axis=1)
    for i in range(data.shape[0]):
        avg[1:, i] = ci(data[i, :])
    return avg

def average_ci(data, axis=1, mode="median"):
    """
    Average data over axis and return average and confidence intervals
    :data: numpy array of n_dim>=2
    Tip: reshape or average data first to apply over multiple axes.
    """
    shape = data.shape
    avg = np.zeros( (3,) + shape[:axis] + shape[axis+1:] )
    if mode == "mean":
        avg[0] = np.nanmean(data, axis=axis)
    elif mode == "median":
        avg[0] = np.nanmedian(data, axis=axis)
    else:
        print("unsupported mode")
    std_err = sem(data, axis=axis, nan_policy="omit")
    avg[1] = avg[0] - std_err * t.ppf((1 + 0.95) / 2, shape[axis] - 1)
    avg[2] = avg[0] + std_err * t.ppf((1 + 0.95) / 2, shape[axis] - 1)
    return avg
