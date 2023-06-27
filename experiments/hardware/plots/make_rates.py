import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    tmin = args.config["options"]["offset"]
    tmax = args.config["options"]["offset"] + args.config["options"]["duration"]
    args.duration = tmax - tmin
    
    shape = (len(args.config["values0"]), len(args.config["values1"]))
    
    rates = np.full(shape, np.nan)
    for i, j in hp.shape_iter(shape):
        spikes = np.load(os.path.join(basedir, args.config["network_spikefiles"][i][j]))
        rates[i,j] = spikes.shape[0] / args.duration / 256.

    rates = hp.average_ci(rates)

    np.savez(args.save, freqs=args.config["values0"], rates=rates, 
             target=args.config["options"]["mu"])
