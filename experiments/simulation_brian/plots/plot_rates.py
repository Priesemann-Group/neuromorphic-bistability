import os
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from functools import partial

from helpers import average_ci
from helpers import shape_iter


def calc_rate(indices, args):
    try:
        data = np.load(args.path.format(
            args.neurons[indices[2]], args.Kext[indices[0]], args.seeds[indices[1]]))
    except FileNotFoundError:
        print("MISSING", indices)
        return np.nan
    spikes = data['spikes']
    time = data['time']
    rate = spikes.shape[0] / (time[-1] - time[0]) / args.neurons
    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--saveto", type=str, default="./plots/")
    parser.add_argument("--neurons", type=int, default=[512], nargs="+")
    parser.add_argument("--seeds", type=int, nargs='+', default=np.arange(10000, 20000, 1000))
    parser.add_argument("--Kext", type=int, nargs='+', default=[66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110])
    args = parser.parse_args()
    print(args.Kext)
    print(args.seeds)

    filename = 'data_brian_N{}_g1.0_Kext{}_seed{}.npz'
    args.path = os.path.join(args.path, filename)
    shape = (len(args.Kext), len(args.seeds), len(args.neurons))

    with mp.Pool() as pool:
        rates = np.array(pool.map(partial(calc_rate, args=args), shape_iter(shape)))

    rates = rates.reshape(shape)
    # rates = average_ci(rates)

    #
    # PLOT
    #
    target = 10
    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    ax = fig.gca()
    # ax.plot(args.Kext, rates[0, :, 0])
    # ax.fill_between(args.Kext, *rates[1:, :, 0], alpha=0.2)
    for j, seed in enumerate(args.seeds):
        ax.plot(args.Kext, rates[:, j, 0], lw=0.1, color="gray")

    ax.axhline(target, linestyle="--", alpha=0.5)
    # ax.set_ylim([0, 2.0 * target])

    ax.set_xlabel(r"average indegree")
    ax.set_ylabel(r"Firing rate (Hz)")

    plt.savefig(f"{args.saveto}figure_N{args.neurons}_rates.pdf", transparent=True)
