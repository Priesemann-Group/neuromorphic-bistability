import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial


def inst_rate(indices, args):
    spikes = np.load(os.path.join(basedir, args.config["network_spikefiles"][indices[0]][indices[1]]))
    counts = np.zeros((256, len(args.bins)-1))
    for neuron in range(counts.shape[0]):
        counts[neuron, :] = np.histogram(spikes[spikes[:,1]==neuron, 0], bins=args.bins)[0] / args.binwidth
    return np.histogram(counts.flatten(), bins=np.arange(0.5e3,500e3,1/args.binwidth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--binwidth", type=float, default=30e-6)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    args.bins = np.arange(0, args.config["options"]["duration"], args.binwidth)

#    shape = (len(args.config["values0"]), len(args.config["values1"]))

#    pool = mp.Pool(20)
#    rates = np.array(pool.map(partial(inst_rate, args=args),
#                              hp.shape_iter(shape)))
#    rates = rates.reshape(shape)
#    rates = hp.median_ci(rates)

#    np.savez(args.save, rates=rates, binwidth=args.binwidth,
#             freqs=args.config["values0"])


    # TODO average over seeds and put plotting in separate script

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, 1, figsize=(3.4, 5*1.6), constrained_layout=True)
    for i, freq in enumerate(args.config['values0']):
        counts, bins = inst_rate((i,0), args)

        ax[i].bar(bins[:-1]/1000, counts, width=1/args.binwidth/1000*0.9, label=f"{freq:.0f}Hz")

        ax[i].set_xlabel(r"Instantaneous firing rate (Hz)")
        ax[i].set_ylabel(r"count")
        ax[i].legend()

    plt.savefig("plots/instantaneous.pdf")
