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


def sort_weights(indices, args):
    try:
        data = np.load(args.path.format(
            args.neurons[indices[2]], args.Kext[indices[0]], args.seeds[indices[1]]))
    except FileNotFoundError:
        print("MISSING", indices)
        return np.full(args.weightbins.size - 1, np.nan)
    exc = data[f'w_{args.type}_exc']
    inh = data[f'w_{args.type}_inh']
    return np.histogram(np.concatenate((exc, -inh)), args.weightbins)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--saveto", type=str, default="./plots/")
    parser.add_argument("--neurons", type=int, default=[512], nargs="+")
    parser.add_argument("--seeds", type=int, nargs='+', default=np.arange(10000, 20000, 1000))
    parser.add_argument("--Kext", type=int, nargs='+', default=[66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110])
    parser.add_argument("--weightbins", type=int, nargs='+', default=np.arange(-64, 64, 1))
    args = parser.parse_args()
    print(args.Kext)
    print(args.seeds)

    filename = 'data_brian_N{}_g1.0_Kext{}_seed{}.npz'
    args.path = os.path.join(args.path, filename)
    shape = (len(args.Kext), len(args.seeds), len(args.neurons))

    with mp.Pool() as pool:
        args.type = "ext"
        weights_ext = np.array(pool.map(partial(sort_weights, args=args),
                               shape_iter(shape)))
        args.type = "rec"
        weights_int = np.array(pool.map(partial(sort_weights, args=args),
                               shape_iter(shape)))

    weights_ext = weights_ext.reshape(shape + (args.weightbins.size - 1, ))
    weights_ext = average_ci(weights_ext)
    weights_int = weights_int.reshape(shape + (args.weightbins.size - 1, ))
    weights_int = average_ci(weights_int)

    #
    # PLOT
    #
    colors = cm.viridis(np.linspace(0, 1, len(args.Kext)))

    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    ax0.set_title(r"(a) Input")
    ax1.set_title(r"(b) Recurrent")

    ax0.set_yscale("log", nonpositive="clip")
    ax1.set_yscale("log", nonpositive="clip")

    for i in range(len(args.Kext)):
        ax0.plot(args.weightbins[:-1], weights_ext[0, i, 0, :], color=colors[i])
        ax1.plot(args.weightbins[:-1], weights_int[0, i, 0, :], color=colors[i])

    ax0.set_xlabel(r"Weight (lsb)")
    ax1.set_xlabel(r"Weight (lsb)")

    ax1.set_yticklabels([])

    ax0.set_ylabel(r"Counts")

    # ax0.set_ylim([1e1, 4e2])
    # ax1.set_ylim([1e1, 4e2])

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=len(args.Kext)),
                           cmap=cm.get_cmap("viridis", len(args.Kext)))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(len(args.Kext)) + 0.5)
    cbar.set_ticklabels(np.round(args.Kext).astype(int))
    cbar.set_label(r"average indegree")

    plt.savefig(f"{args.saveto}figure_N{args.neurons}_weights.pdf", transparent=True)
