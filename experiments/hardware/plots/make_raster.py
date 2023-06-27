import os
import argparse
import json
import helpers as hp
import numpy as np


TMIN = 0.005
TMAX = 0.006


def filter_spiketrain(data):
    mask = np.logical_and(data[:, 0] >= TMIN, data[:, 0] < TMAX)
    data = data[mask, :]
    data[:, 0] -= TMIN
    return data

def activity(data, bins):
    return np.histogram(data[:, 0], bins=bins)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--binwidth", type=float, default=1e-6)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()
    
    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    args.bins = np.arange(0, args.config["options"]["duration"], args.binwidth)
    
    shape = (len(args.config["values0"]), len(args.config["values1"]))
    
    for f, freq in enumerate(args.config["values0"]):
        for s, seed in enumerate(args.config["values1"]):
            times = np.load(os.path.join(basedir, args.config["network_spikefiles"][f][s]))[:, 0]

    freqs = np.array([args.config["values0"][0], args.config["values0"][-1]])

    low = np.load(os.path.join(basedir, args.config["network_spikefiles"][1][2]))
    high = np.load(os.path.join(basedir, args.config["network_spikefiles"][-1][2]))

    bins = np.arange(TMIN, TMAX + args.binwidth, args.binwidth)

    np.savez(args.save,
             freqs=freqs, bins=bins,
             raster_low=filter_spiketrain(low),
             raster_high=filter_spiketrain(high),
             activity_low=activity(low, bins),
             activity_high=activity(high, bins))
