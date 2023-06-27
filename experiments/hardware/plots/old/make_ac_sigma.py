import data
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

def func(x, A, tau, o):
    return np.abs(A) * np.exp(-x / tau) + o

def coefficients(indices, conf):
    sigma = conf.sigmas[indices[1]]
    seed = conf.seeds[indices[2]]
    spikes = np.load(conf.paths[indices[0]].format(sigma, seed))
    activity = np.histogram(spikes[:, 0], bins=conf.bins)[0]
    
    rk = np.zeros(conf.Nsteps)
    for i, step in enumerate(conf.steps):
        front = activity[:-step] - activity[:-step].mean()
        back = activity[step:] - activity[step:].mean()

        rk[i] = np.mean(front * back) / activity[:-step].var()
    return rk

def fit(rk, steps):
    tau0 = 100
    a0 = rk[0] / np.exp(-steps[0] / tau0)
    o0 = rk[-1]
    p0 = np.array([a0, tau0, o0])
    try:
        return curve_fit(func, steps, rk, p0=p0)[0][1]
    except:
        return np.nan

def average_rks(data):
    avg = np.zeros((3, data.shape[0], data.shape[1], data.shape[3]))
    avg[0, :] = np.nanmedian(data, axis=2)
    for i, j, k in hp.shape_iter((data.shape[0], data.shape[1], data.shape[3])):
        avg[1:, i, j, k] = hp.ci(data[i, j, :, k])
    return avg

def average_taus(data):
    avg = np.zeros((3, data.shape[0], data.shape[1]))
    avg[0, :] = np.nanmedian(data, axis=2)
    for i, j in hp.shape_iter((data.shape[0], data.shape[1])):
        avg[1:, i, j] = hp.ci(data[i, j, :])
    return avg


if __name__ == "__main__":
    conf = data.Sigma()

    shape = (conf.Npaths, conf.Nsigmas, conf.Nseeds)

    pool = mp.Pool(8)
    rks = np.array(pool.map(partial(coefficients, conf=conf),
                            hp.shape_iter(shape)))

    taus = np.array(pool.map(partial(fit, steps=conf.steps), rks))
    
    rks = rks.reshape(shape + (conf.Nsteps, ))
    rks = average_rks(rks)
    
    taus = taus.reshape(shape)
    taus = average_taus(taus)
    taus *= conf.binwidth * conf.speedup
    
    np.savez("data/ac_sigma.npz", rks=rks, taus=taus)
