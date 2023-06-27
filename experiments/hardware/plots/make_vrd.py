import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial
from scipy.stats import sem, t
from scipy.optimize import curve_fit

from make_ac import func

NEURONS = 256


def average(data):
    avg = np.full((3, data.shape[0], data.shape[2], data.shape[5]), np.nan)
    avg[0, :, :] = np.nanmean(data, axis=(1, 3, 4))

    # for i, j, k in hp.shape_iter((data.shape[0], data.shape[2], data.shape[5])):
    #     avg[1:, i, j, k] = hp.ci(data[i, :, j, :, :, k])

    dim = data.shape[1] * data.shape[3] * data.shape[4]
    for i, j, k in hp.shape_iter((data.shape[0], data.shape[2], data.shape[5])):
        d = data[i, :, j, :, :, k]
        d = d[~np.isnan(d)]
        std_err = sem(d.flatten())
        avg[1, i, j, k] = avg[0, i] - std_err * t.ppf((1 + 0.95) / 2, dim - 1)
        avg[2, i, j, k] = avg[0, i] + std_err * t.ppf((1 + 0.95) / 2, dim - 1)
        # avg[1:, i, j] = hp.ci(data[i, :, j, :, :, k])
    return avg

def distance(lhs, rhs, sigma):
    if lhs.sum() or rhs.sum():
        return ((lhs - rhs)**2).sum() / float(sigma*((lhs + rhs)**2).sum())
    else:
        return 0

def main(indices, config):
    print(indices)
    path = os.path.join(args.basedir,
                        config["network_spikefiles"][indices[0]][indices[1]])
    
    timebins = np.arange(-args.window, 0, args.binwidth)

    mat = np.zeros((config["classes"], config["samples"], NEURONS,
                    bins.size, timebins.size-1))
    vrd = np.full((2, config["classes"], config["samples"], bins.size), np.nan)

    steps = int(3 * args.sigma / args.binwidth)
    kerneltimes = (np.arange(-steps, steps + 1, 1) * args.binwidth).astype(float)
    kernel = np.exp(-(kerneltimes / args.sigma)**2 / 2.0)
    try:
        spikes = np.load(path, allow_pickle=True)
        for c in range(config["classes"]):
            for s in range(config["samples"]):
                for n in range(NEURONS):
                    mask = (spikes[c][s][:, 1] == n)
                    tmp = spikes[c][s][mask, 0]
                    for b in range(bins.size):
                        a = np.histogram(spikes[c][s][mask, 0],
                                         bins=timebins+bins[b])[0]
                        mat[c, s, n, b, :] = np.convolve(a, kernel, "same")

        # difference between samples
        for c in range(config["classes"]):
            for s in range(config["samples"] - 1):
                for b in range(bins.size):
                    vrd[0, c, s, b] = distance(mat[c, s,   :, b, :].flatten(),
                                               mat[c, s+1, :, b, :].flatten(),
                                               args.sigma)

        # difference between classes
        for c in range(config["classes"] - 1):
            for s in range(config["samples"]):
                for b in range(bins.size):
                    vrd[1, c, s, b] = distance(mat[c,   s, :, b, :].flatten(),
                                               mat[c+1, s, :, b, :].flatten(),
                                               args.sigma)
    except IOError:
        print("File {} not found!".format(path))
    return vrd

def fit(tmp, bins):
    tau0 = 20
    a0 = tmp[0] / np.exp(-bins[0] / tau0)
    o0 = tmp[-1]
    p0 = np.array([a0, tau0, o0])
    try:
        return curve_fit(func, bins, tmp, p0=p0)[0][1]
    except:
        return np.nan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--binwidth", type=float, default=1e-6)
    parser.add_argument("--window", type=float, default=100e-6)
    parser.add_argument("--step", type=float, default=10e-6)
    parser.add_argument("--sigma", type=float, default=10e-6)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    args.basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    values0 = args.config["values0"]
    values1 = args.config["values1"]

    shape = (len(values0), len(values1))
 
    start = args.config["pre_letter"] - 2
    end = args.config["pre_letter"] + args.config["post_letter"] + 1
    bins = np.arange(args.config["options"]["offset"] + start * args.config["duration"],
                     args.config["options"]["offset"] +   end * args.config["duration"] - args.window,
                     args.step)
    
    boarders = list()
    boarders.append(args.config["options"]["offset"] + (args.config["pre_letter"] + 0) * args.config["duration"])
    boarders.append(args.config["options"]["offset"] + (args.config["pre_letter"] + 1) * args.config["duration"])

    with mp.Pool(20) as pool:
        vrd = np.array(pool.map(partial(main, config=args.config),
                       hp.shape_iter(shape)))
    vrd = vrd.reshape(shape + (2, args.config["classes"], args.config["samples"], bins.size))
    vrd = average(vrd)

    tmp = vrd[0, :, 1, :] - vrd[0, :, 0, :]
    mask = bins > (args.config["pre_letter"] + 2) * args.config["duration"]

    with mp.Pool(20) as pool:
        decay = np.array(pool.map(partial(fit, bins=bins[mask]-bins[mask][0]),
                                  tmp[:, mask]))

    np.savez(args.save,
             vrd=vrd, decay=decay*args.step,
             values0=values0, bins=bins, boarders=boarders)
