import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial


def estimate_mi_ext(indices, args):
    print(indices)
    spikes1 = np.load(os.path.join(basedir, args.config["input_spikefiles"][indices[0]][indices[1]]))
    spikes2 = np.load(os.path.join(basedir, args.config["network_spikefiles"][indices[0]][indices[1]]))
    
    mi = np.full((256, 256, args.delays.size), np.nan)
    for i, pre in enumerate(np.random.choice(256, size=20, replace=False)):
        spk1 = spikes1[(spikes1[:, 1] == pre), 0]
        act1 = (np.histogram(spk1, bins=args.bins)[0] > 0).astype(np.int)
        for j, post in enumerate(np.random.choice(256, size=20, replace=False)):
            spk2 = spikes2[(spikes2[:, 1] == pre), 0]
            act2 = (np.histogram(spk2, bins=args.bins)[0] > 0).astype(np.int)
            for k, d in enumerate(args.delays):
                mi[i, j, k] = hp.mutual_info(act1[:-d], act2[d:])
    return mi

def estimate_mi_int(indices, args):
    print(indices)
    spikes = np.load(os.path.join(basedir, args.config["network_spikefiles"][indices[0]][indices[1]]))
    
    mi = np.full((256, 256, args.delays.size), np.nan)
    for i, pre in enumerate(np.random.choice(256, size=20, replace=False)):
        spk1 = spikes[(spikes[:, 1] == pre), 0]
        act1 = (np.histogram(spk1, bins=args.bins)[0] > 0).astype(np.int)
        for j, post in enumerate(np.random.choice(256, size=50, replace=False)):
            spk2 = spikes[(spikes[:, 1] == pre), 0]
            act2 = (np.histogram(spk2, bins=args.bins)[0] > 0).astype(np.int)
            for k, d in enumerate(args.delays):
                mi[i, j, k] = hp.mutual_info(act1[:-d], act2[d:])
    return mi

def average(data):
    avg = np.zeros((3, data.shape[0], data.shape[4]))
    avg[0, :] = np.nanmedian(data, axis=(1, 2, 3))
    for i, j in hp.shape_iter((data.shape[0], data.shape[4])):
        avg[1:, i, j] = hp.ci(data[i, :, :, :, j].flatten())
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--binwidth", type=float, default=5e-6)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    tmin = args.config["options"]["offset"]
    tmax = args.config["options"]["offset"] + args.config["options"]["duration"]
    args.bins = np.arange(tmin, tmax, args.binwidth)

    args.delays = np.logspace(np.log10(1), np.log10(100), 20, dtype=np.int)
    args.delays = np.unique(args.delays)
 
    shape = (len(args.config["values0"]), len(args.config["values1"]))

    with mp.Pool(20) as pool:
        mi_ext = np.array(pool.map(partial(estimate_mi_ext, args=args),
                                   hp.shape_iter(shape)))
        mi_ext = mi_ext.reshape(shape + (256, 256, args.delays.size, ))
        mi_ext = average(mi_ext)
    
        mi_int = np.array(pool.map(partial(estimate_mi_int, args=args),
                                   hp.shape_iter(shape)))

    mi_int = mi_int.reshape(shape + (256, 256, args.delays.size, ))
    mi_int = average(mi_int)
    
    delays = args.delays * args.binwidth
    
    np.savez(args.save,
             freqs=args.config["values0"], delays=delays,
             mi_ext=mi_ext, mi_int=mi_int)
