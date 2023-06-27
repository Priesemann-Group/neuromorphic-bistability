import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial

NEURONS = 256

def main(indices, config):
    path = os.path.join(basedir, config["input_spikefiles"][indices[0]][indices[1]])
    activity = np.full((config["classes"], config["samples"], args.bins.size - 1), np.nan)
    try:
        spikes = np.load(path, allow_pickle=True)
        for c in range(config["classes"]):
            for s in range(config["samples"]):
                spikes[c][s][:,0] -= config["pre_letter"]*config["duration"] + config["options"]["offset"]
                activity[c, s, :] = np.histogram(spikes[c][s][:, 0], bins=args.bins)[0]
    except IOError:
        print("File {} not found!".format(path))
    return np.nanmean(activity, axis=(0,1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--binwidth", type=float, default=1e-6)
    args = parser.parse_args()
    args.config = "/wang/users/bcramer/cluster_home/project/paranoise/experiments/hardware/data/letter_new/sweep_calib_config.json"
    print(args.config)

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    # letter = args.config["pre_letter"] + args.config["post_letter"] + 1
    # bins = np.arange( (config["pre_letter"]-3) * config["duration"],
    #                   (letter+1)*config["duration"],
    #                   args.binwidth) + config["options"]["offset"]

    args.bins = np.arange(-1 * args.config["duration"],
                          (0+2) * args.config["duration"],
                          args.binwidth)

    shape = (len(args.config["values0"]), len(args.config["values1"]))
    print(shape)

    with mp.Pool(20) as pool:
        act = np.array(pool.map(partial(main, config=args.config),
                      hp.shape_iter(shape)))
    act = act.reshape(shape + (-1, ))
    act = hp.average_ci(act, mode="mean")
    act /= args.binwidth*1000*NEURONS

    np.savez(args.save,
            act=act,
            values0=args.config["values0"],
            bins=args.bins/args.config["duration"])
