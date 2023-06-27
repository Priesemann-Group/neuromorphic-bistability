import argparse
import os
import json
import numpy as np
import helpers as hp
from scipy.stats import sem, t


def average(data):
    avg = np.zeros((3, data.shape[0], data.shape[2]))
    avg[0, :, :] = np.nanmean(data, axis=1)
    for i, j in hp.shape_iter((data.shape[0], data.shape[2])):
        d = data[i, :, j]
        d = d[~np.isnan(d)]
        std_err = sem(d.flatten())
        avg[1, i] = avg[0, i] - std_err * t.ppf((1 + 0.95) / 2, data.shape[1] - 1)
        avg[2, i] = avg[0, i] + std_err * t.ppf((1 + 0.95) / 2, data.shape[1] - 1)
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    shape = (len(args.config["values0"]), len(args.config["values1"]))
    print(shape)
    bins = np.arange(-64, 64, 1)

    weights_ext = np.full(shape + (bins.size-1, ), np.nan)
    weights_int = np.full(shape + (bins.size-1, ), np.nan)

    sparsity = np.full(shape, np.nan)
    counts = np.full(shape, np.nan)

    for i, j in hp.shape_iter(shape):
        weights = np.load(os.path.join(basedir, args.config["weightfiles"][i][j]))
        sparsity[i, j] = (weights["weights"] > 0).sum() / (256. * 512.)
        counts[i, j] = (weights["weights"] == 63).sum()
        inh_mask = weights["inh_mask"]
        inh_mask = -1 * (2 * inh_mask - 1)
        weights = weights["weights"] * inh_mask[:,None]

        weights_ext[i, j, :] = np.histogram(weights[256:, :], bins)[0]
        weights_int[i, j, :] = np.histogram(weights[:256, :], bins)[0]

    print(sparsity.mean(axis=1))
    print(counts.mean(axis=1))
    np.savez("correlation.npz", sparsity=sparsity, counts=counts)

    weights_ext = hp.average_ci(weights_ext)
    weights_int = hp.average_ci(weights_int)

    weights_ext[:,:,64] = np.nan
    weights_int[:,:,64] = np.nan

    np.savez(args.save,
             weights_ext=weights_ext, weights_int=weights_int,
             bins=bins, freqs=args.config["values0"])
