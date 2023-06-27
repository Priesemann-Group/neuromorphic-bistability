import os
import argparse
import json
import numpy as np
import multiprocessing as mp
from functools import partial

import helpers as hp

def estimate_pert(indices, args):
    i, j = indices
    if j==0: print(i)
    activity = np.full((256, args.bins.size - 1), np.nan)
    try:
        spikes = np.load(os.path.join(basedir,
                                      args.config["network_spikefiles"][i][j]),
                                      allow_pickle=True)
        for k, spks in enumerate(spikes):
            activity[k, :] = np.histogram(spks[:, 0], bins=args.bins)[0]
    except IOError:
        pass
    return activity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--binwidth", type=float, default=2e-6)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    args.bins = np.arange(args.config["perttime"] - 200e-6,
                          args.config["perttime"] + 300e-6,
                          args.binwidth) #+ args.config["options"]["offset"]

    shape = (len(args.config["values0"]), len(args.config["values1"]))
    print(shape)

    with mp.Pool(20) as pool:
        pert = np.array(pool.map(partial(estimate_pert, args=args),
                                 hp.shape_iter(shape)))
    # TODO calculate CI over neurons as well
    shape = (shape[0], shape[1]*256)
    pert = pert.reshape(shape + (args.bins.size - 1,))
    #pert = np.nanmean(pert, axis=2)
    pert = hp.average_ci(pert)

    np.savez(args.save, freqs=args.config["values0"], pert=pert,
             bins=args.bins-args.config["perttime"])
