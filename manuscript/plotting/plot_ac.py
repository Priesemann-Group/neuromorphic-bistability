import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    dts = data["dts"] * 1000.
    rks = data["rks"]
    taus = data["taus"] * 1000.

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.4, 1.3))
    ax = fig.gca()
    axi = inset_axes(ax, width="60%", height="45%", loc="upper right")

    axi.set_yscale("log", nonpositive="mask")

    for i in np.arange(h.size)[1:][::2]:
        ax.plot(dts, rks[0, i, :], color=colors[i])
        ax.fill_between(dts,
                        rks[0, i, :] - rks[1, i, :],
                        rks[0, i, :] + rks[1, i, :],
                        alpha=0.2, facecolor=colors[i])

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.5, 0.5))

    for i in range(h.size):
        axi.errorbar(h[i], taus[0, i],
                     yerr=taus[1, i],
                     color=colors[i], fmt="x", ms=2.0, fillstyle="none",
                     linewidth=0.8, markeredgewidth=0.6)

    axi.set_xticks([0.6, 0.8, 1.0])
    axi.set_ylim(3e-3, 9.5e-1)

    axi.set_xlabel(r"$h$ (\si{\kilo\hertz})", fontsize=6)
    axi.set_ylabel(r"$\tau_\mathrm{AC}$ (\si{\second})", fontsize=6, rotation="horizontal")

    ax.set_xlabel(r"Time lag $t'$ (\si{\second})")
    ax.set_ylabel(r"Correlation $C(t')$\hphantom{C}")

    axi.yaxis.set_label_coords(-0.15,1.08)

    for tick in axi.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in axi.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    plt.savefig(args.save)
