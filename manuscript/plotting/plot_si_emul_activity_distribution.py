import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    distribution_bins = data["distribution_bins"] * 1e-3
    distribution = data["distribution"] * 1e3

    # This can be used to extract the data for the fits
    # data = np.zeros((distribution_bins.size - 1, distribution.shape[0] + 1))
    # data[:, 0] = distribution_bins[:-1]
    # data[:, 1:] = distribution.T
    # np.savetxt("activity_distributions.txt", data)

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.9)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.2, 1.3))
    ax = fig.gca()
    axi = inset_axes(ax, width="50%", height="35%", loc="upper right")

    for i in np.arange(h.size)[::2]:
        ax.plot(distribution_bins[:-1], distribution[i, :],
                color=colors[i])

    ax.set_xlabel(r"Rate $\nu$ (\si{\hertz})")
    ax.set_ylabel(r"Probability $P(\nu)$\hphantom{C}")

    ax.set_ylim(0, 0.04)
    ax.set_xlim(0, 150)
    ax.set_yticks(np.arange(0, 0.05, 0.01))

    for i in np.arange(h.size)[::2]:
        axi.plot(distribution_bins[:-1], distribution[i, :],
                 color=colors[i], linewidth=0.8, zorder=h.size - i)

    axi.set_yscale("log", nonpositive="clip")

    axi.set_xlabel(r"$\nu$ (\si{\hertz})", fontsize=6)
    axi.set_ylabel(r"$P(\nu)$", rotation="horizontal", fontsize=6)

    axi.yaxis.set_label_coords(-0.15,1.08)

    for tick in axi.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in axi.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    axi.set_ylim(3e-4, 9.5e-2)
    axi.set_xlim(0, 120)

    plt.savefig(args.save, transparent=True)
