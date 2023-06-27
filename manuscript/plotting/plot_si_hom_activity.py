import argparse
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    h_selection = data["h_selection"] * 1e-6
    bins = data["bins"][:-1] * 1e3
    mask = np.logical_and(bins >= 21.0, bins < 24.0)
    bins = bins[mask] - 21.0
    activity_high = data["activity_high"][mask] * 1e-3
    activity_intermediate = data["activity_intermediate"][mask] * 1e-3
    activity_low = data["activity_low"][mask] * 1e-3

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.9)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])

    colors = cmap(norm(h_selection))

    fig = plt.figure(figsize=(1.4, 1.3))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35)

    ax2 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0 = fig.add_subplot(gs[2])
    fax = fig.add_subplot(gs[:])

    ax0.text(2.9, 150, r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(h_selection[0]), ha="right", fontsize=6)
    ax0.plot(bins, activity_high, color=colors[0])
    ax1.text(2.9, 150, r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(h_selection[1]), ha="right", fontsize=6)
    ax1.plot(bins, activity_intermediate, color=colors[1])
    ax2.text(2.9, 150, r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(h_selection[2]), ha="right", fontsize=6)
    ax2.plot(bins, activity_low, color=colors[2])

    ax0.set_xlim(0, 3)
    ax1.set_xlim(0, 3)
    ax2.set_xlim(0, 3)
    ax0.set_ylim(0, 200)
    ax1.set_ylim(0, 200)
    ax2.set_ylim(0, 200)

    ax2.set_xticklabels([])
    ax1.set_xticklabels([])

    ax0.set_yticks(np.arange(0, 300, 100))
    ax1.set_yticks(np.arange(0, 300, 100))
    ax2.set_yticks(np.arange(0, 300, 100))

    fax.spines["right"].set_visible(False)
    fax.spines["left"].set_visible(False)
    fax.spines["bottom"].set_visible(False)
    fax.spines["top"].set_visible(False)

    fax.set_xticks([])
    fax.set_yticks([])
    fax.set_ylabel(r"Rate $\nu$ (\si{\hertz})")
    fax.set_xlabel(r"Time $t$ (\si{\second})")
    fax.xaxis.set_label_coords(0.5, -0.18)
    fax.yaxis.set_label_coords(-0.32, 0.5)

    plt.savefig(args.save, transparent=True)
