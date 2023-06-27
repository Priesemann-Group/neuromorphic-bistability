import os
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
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
        return np.full((1000,), np.nan)
    rates = data['nu']
    return rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--saveto", type=str, default="./plots/")
    parser.add_argument("--neurons", type=int, default=[512], nargs="+")
    parser.add_argument("--seeds", type=int, nargs='+', default=[10000])
    parser.add_argument("--Kext", type=int, nargs='+', default=[66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110])
    args = parser.parse_args()
    print(args.Kext)
    print(args.seeds)

    filename = 'data_brian_N{}_g1.0_Kext{}_seed{}.npz'
    args.path = os.path.join(args.path, filename)
    shape = (len(args.Kext), len(args.seeds), len(args.neurons))

    with mp.Pool() as pool:
        rates = np.array(pool.map(partial(calc_rate, args=args), shape_iter(shape)))

    rates = rates.reshape(shape + (-1,))
    rates = average_ci(rates)

    #
    # PLOT
    #
    target = 10
    colors = cm.viridis(np.linspace(0, 1, len(args.Kext)))
    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.96, 0.04])

    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    # ax.set_xscale("log", nonposx="clip")

    x = np.arange(rates.shape[-1])
    for i, kin in enumerate(args.Kext):
        ax.plot(x, rates[0, i, 0, :], color=colors[i])
        ax.fill_between(x, *rates[1:, i, 0, :], alpha=0.2, color=colors[i])
        # for j, seed in enumerate(args.seeds):
        #     if True:  # np.all(rates[i, j, -10:] > 5):
        #         ax.plot(x, rates[i, j, :], color=colors[i], lw=0.2, alpha=0.2)
        #     else:
        #         print("bad seed", kin, seed)

    ax.axhline(target, linestyle="--", alpha=0.5)
    # ax.set_ylim([0, 2.0 * target])
    ax.set_ylim([0, 100])

    ax.set_xlabel(r"update")
    ax.set_ylabel(r"Firing rate (Hz)")

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=len(args.Kext)),
                           cmap=cm.get_cmap("viridis", len(args.Kext)))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(len(args.Kext)) + 0.5)
    cbar.set_ticklabels(np.round(args.Kext, 1))
    cbar.set_label(r"average indegree")

    plt.savefig(f"{args.saveto}figure_N{args.neurons}_development.pdf", transparent=True)
