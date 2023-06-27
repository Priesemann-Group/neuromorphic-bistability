import os
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from functools import partial

from helpers import average_ci
from helpers import shape_iter

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False


def coefficients(indices, args):
    try:
        data = np.load(args.path.format(
            args.neurons[indices[2]], args.Kext[indices[0]], args.seeds[indices[1]]))
    except FileNotFoundError:
        print("MISSING", indices)
        return np.full(args.ratebins.size - 1, np.nan)
    spikes = data["spikes"]
    time = data['time']
    bins = np.arange(time[0], time[-1] + time[1] - time[0], args.binwidth)
    # Brian output of spikes is in native ms scale but time is in seconds ...
    activity = np.histogram(spikes[:, 0] / 1000, bins=bins)[0]
    counts = np.histogram(activity, bins=args.ratebins)[0]
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--saveto", type=str, default="./plots/")
    parser.add_argument("--neurons", type=int, default=[256, 512, 786, 1024], nargs="+")
    parser.add_argument("--seeds", type=int, nargs='+', default=np.arange(567800, 567850, 1))
    parser.add_argument("--Kext", type=int, nargs='+', default=[70, 90, 110])
    parser.add_argument("--binwidth", type=float, default=5e-3)
    parser.add_argument("--ratebins", type=int, nargs='+', default=np.arange(0, 500, 1))
    args = parser.parse_args()
    print(args.Kext)
    print(args.seeds)

    filename = 'data_brian_N{}_g1.0_Kext{}_seed{}.npz'
    args.path = os.path.join(args.path, filename)
    shape = (len(args.Kext), len(args.seeds), len(args.neurons))

    with mp.Pool() as pool:
        counts = np.array(pool.map(partial(coefficients, args=args),
                                   shape_iter(shape)))

    counts = counts.reshape(shape + (args.ratebins.size - 1, ))
    counts = counts.sum(axis=1)
    counts = counts / counts.sum(axis=2)[:, :, None]

    np.savez("simulation_activity_distribution.npz",
             distribution=counts * args.binwidth,
             distribution_bins=args.ratebins / args.binwidth,
             h=np.array(args.Kext) * 10.,
             neurons=args.neurons)

    #
    # PLOT
    #
    colors = cm.viridis(np.linspace(0, 1, len(args.Kext)))

    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.96, 0.04])

    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    # ax.set_yscale("log", nonpositive="clip")

    handles = []
    # linestyles = ["-", ":", "--", "-."]
    linestyles = ["-", "-", "-", "-"]
    alphas = [1, 0.75, 0.5, 0.25]
    for i, kext in enumerate(args.Kext):
        for j, N in enumerate(args.neurons):
            ax.plot(args.ratebins[:-1], np.log(counts[i, j, :]), color=colors[i], linestyle=linestyles[j], alpha=alphas[j])
            # ax.fill_between(args.ratebins[:-1], *np.log(counts[1:, i, j, :]), color=colors[i], alpha=0.2)

            if i == 0:
                handles.append(mlines.Line2D([], [], color='gray', alpha=alphas[j], label=args.neurons[j], linestyle=linestyles[j]))

    ax.legend(loc="lower right", handles=handles)

    ax.set_xlabel(r"rate (Hz)")
    ax.set_ylabel(r"log(counts)")

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=len(args.Kext)),
                           cmap=cm.get_cmap("viridis", len(args.Kext)))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(len(args.Kext)) + 0.5)
    cbar.set_ticklabels(np.round(args.Kext, 1))
    cbar.set_label(r"average indegree")

    plt.savefig(f"{args.saveto}figure_N{args.neurons}_histogram.pdf", transparent=True)
