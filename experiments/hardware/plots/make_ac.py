import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial
from scipy.optimize import curve_fit


def func(x, A, tau, o):
    return np.abs(A) * np.exp(-x / tau) + o

def coefficients(indices, args):
    spikes = np.load(os.path.join(basedir, args.config["network_spikefiles"][indices[0]][indices[1]]))

    activity = np.histogram(spikes[:, 0], bins=args.bins)[0]
    
    rk = np.zeros(args.steps.size)
    for i, step in enumerate(args.steps):
        front = activity[:-step] - activity[:-step].mean()
        back = activity[step:] - activity[step:].mean()

        rk[i] = np.mean(front * back) / activity[:-step].var()
    return rk

def fit(rk, steps):
    tau0 = 50
    a0 = rk[0] / np.exp(-steps[0] / tau0)
    o0 = rk[-1]
    p0 = np.array([a0, tau0, o0])
    try:
        return curve_fit(func, steps, rk, p0=p0)[0][1]
    except:
        return np.nan

def average_rks(data):
    avg = np.zeros((3, data.shape[0], data.shape[2]))
    avg[0, :] = np.nanmedian(data, axis=1)
    for i, j in hp.shape_iter((data.shape[0], data.shape[2])):
        avg[1:, i, j] = hp.ci(data[i, :, j])
    return avg

def average_taus(data):
    avg = np.zeros((3, data.shape[0]))
    avg[0, :] = np.nanmedian(data, axis=1)
    for i in range(data.shape[0]):
        avg[1:, i] = hp.ci(data[i, :])
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--binwidth", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    tmin = args.config["options"]["offset"]
    tmax = args.config["options"]["offset"] + args.config["options"]["duration"]
    args.bins = np.arange(tmin, tmax, args.binwidth)
    args.steps = np.arange(1, args.steps)
    
    shape = (len(args.config["values0"]), len(args.config["values1"]))

    with mp.Pool(20) as pool:
        rks = np.array(pool.map(partial(coefficients, args=args),
                                hp.shape_iter(shape)))

        taus = np.array(pool.map(partial(fit, steps=args.steps), rks))
    
    rks = rks.reshape(shape + (args.steps.size, ))
    rks = hp.average_ci(rks)
    
    taus = taus.reshape(shape)
    taus = hp.average_ci(taus)
    taus *= args.binwidth

    dts = args.steps * args.binwidth
    
    np.savez(args.save,
             freqs=args.config["values0"], dts=dts, rks=rks, taus=taus)
