from itertools import product
import numpy as np
from scipy.stats import t


def average_se(data, axis=1):
    N = data.shape[axis]
    avg = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis)
    r = np.empty((2, ) + avg.shape)
    r[0] = avg
    r[1] = std / np.sqrt(N)
    return r


def average_ci(data, axis=1, mode="median"):
    """
    Average data over axis and return average and confidence intervals
    :data: numpy array of n_dim>=2
    :axis: int or tuple
    :mode: mean or median
    """
    # using the corrected for small N answer
    # https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
    shape = data.shape
    # TODO: np.nan should not be counted in N
    if isinstance(axis, int):
        N = shape[axis]
    elif isinstance(axis, tuple):
        N = np.prod([shape[x] for x in axis])
    else:
        raise TypeError("axis must be int or tuple")

    if mode == "mean":
        avg = np.nanmean(data, axis=axis)
    elif mode == "median":
        avg = np.nanmedian(data, axis=axis)
    else:
        raise TypeError("mode must be mean or median")

    r = np.empty((3,) + avg.shape)
    r[0] = avg

    # corrected sample standard deviation, still biased because of sqrt taken
    # was sem from scipy, which does not support tuple axis
    # std_err = sem(data, axis=axis, nan_policy="omit") \
    #     * t.ppf((1 + 0.95) / 2, shape[axis] - 1)
    std_err = np.std(data, axis=axis, ddof=1) \
        / np.sqrt(N) \
        * t.ppf((1 + 0.95) / 2, N - 1)

    if mode == "median":
        # https://en.wikipedia.org/wiki/Median#Efficiency
        std_err *= np.sqrt(np.pi / 2)

    r[1] = avg - std_err
    r[2] = avg + std_err
    return r


def shape_iter(shape):
    return product(*[range(n) for n in shape])
