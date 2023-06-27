import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    w_inh = data["w_inh"]
    w_exc = data["w_exc"]
    h = data["h"][::-1] * 1e-6
    bins = data["bins"] * 1e-3
    distribution = data["distribution"] * 1e3

    norm = colors.TwoSlopeNorm(vmin=h[2], vmax=h[-1], vcenter=0.9)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    c = cmap(norm(h[2:]))
    colors = np.zeros((c.shape[0] + 2, c.shape[1]))
    colors[0, :] = np.array([0.0, 0.0, 0.0, 1.0])
    colors[1, :] = np.array([0.5, 0.5, 0.5, 1.0])
    colors[2:, :] = c

    fig = plt.figure(figsize=(1.4, 1.3))
    ax = fig.gca()

    ax.set_yscale("log")
    ax.text(0.95, 0.95,
            r"$g={}/{}$".format(w_inh, w_exc),
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            transform=ax.transAxes)

    for i in np.arange(h.size):
        ax.plot(bins[:-1], distribution[h.size - (i + 1), :],
                color=colors[i])

    ax.set_xlabel(r"Rate $\nu$ (\si{\hertz})")
    ax.set_ylabel(r"Probability $P(\nu)$\hphantom{C}")

    ax.set_ylim(1e-6, 1e0)
    ax.set_xlim(0, 150)
    ax.yaxis.set_label_coords(-0.32, 0.5)

    plt.savefig(args.save, transparent=True)
