import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial


NEURONS = 256

def average(data):
    avg = np.full((3, data.shape[0]), np.nan)
    avg[0, :, :] = np.nanmedian(data, axis=1)
    for i in hp.shape_iter((data.shape[0])):
        avg[1:, i] = hp.ci(data[i, :])
    return avg

def main(indices, config):
    print(indices)
    path = config["input_spikefiles"][indices[0]][indices[1]]
    
    counts = 0
    try:
        spikes = np.load(path, allow_pickle=True)
        for cla in spikes:
            for i in range(len(cla)):
                for j in range(len(cla)):
                    if cla[i].shape != cla[j].shape:
                        counts += 1
    except IOError:
        print("File {} not found!".format(path))
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=argparse.FileType("r"))
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    args.config = json.load(args.config)
    
    values0 = args.config["values0"]
    values1 = args.config["values1"]

    shape = (len(values0), len(values1))

    pool = mp.Pool(10)
    mi = np.array(pool.map(partial(main, config=args.config),
                  hp.shape_iter(shape)))
    mi = mi.reshape(shape)
    np.save(args.save, mi)
