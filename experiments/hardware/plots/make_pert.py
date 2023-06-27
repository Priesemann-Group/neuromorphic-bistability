import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial

def estimate_pert(indices, args):
    i, j = indices
    if j==0: print(i)
    diff = np.full(256, np.nan)
    try:
        spikes = np.load(os.path.join(basedir, args.config["network_spikefiles"][i][j]),
                         allow_pickle=True)
        for k, spks in enumerate(spikes):
            activity = np.histogram(spks[:, 0], bins=args.bins)[0]
            diff[k] = np.diff(activity)
    except IOError as e:
        print("misssing",i,j)
        pass
    except Exception as e:
        print(i, j, repr(e))
        pass

    return diff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--binwidth", type=float, default=250e-6)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    perttime = args.config["perttime"] + 1e-6 #+ args.config["options"]["offset"]
    args.bins = [perttime-args.binwidth,
                 perttime,
                 perttime+args.binwidth]

    shape = (len(args.config["values0"]), len(args.config["values1"]))
    print(shape)

    with mp.Pool(20) as pool:
        pert = np.array(pool.map(partial(estimate_pert, args=args),
                                 hp.shape_iter(shape)))
    pert = pert.reshape(shape + (256, ))
    # average pert over neurons
    pert = np.nanmean(pert, axis=2)
    pert = hp.average_ci(pert)

    np.savez(args.save, freqs=args.config["values0"], pert=pert)
