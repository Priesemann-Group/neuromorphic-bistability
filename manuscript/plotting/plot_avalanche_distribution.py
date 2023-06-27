import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from scipy.optimize import curve_fit


def func(s, A, sstar):
    return A * np.exp(-s / sstar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    h_selection = data["h_selection"] * 1e-6
    params = data["params"]
    edges = data["edges"]
    counts = data["counts"]

    swap = data["swap"].astype(bool)
    activity = data["activity"]
    transmat = data["transmat"]

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(3.5, 1.9))
    ax = fig.gca()

    ax.set_xscale("log", nonpositive="clip")
    ax.set_yscale("log", nonpositive="clip")

    x = np.logspace(np.log10(1e0), np.log10(1e6))
    for i, htmp in enumerate(h_selection):
        pos = ~swap[i]
        sstar = -1 * activity[i] / np.log(transmat[i, int(pos), int(pos)])
        ps = np.exp(-1 * edges[i, :] / sstar)

        mask = edges[i, :] > 1e3

        A = curve_fit(lambda s, A: func(s, A, sstar), edges[i, mask], counts[i, mask], p0=1.0, maxfev=10000)[0][0]

        ax.plot(x, func(x, A, sstar), color=colors[i], linestyle=":", alpha=0.7, zorder=-1)
        ax.plot(edges[i, :], counts[i, :],
                marker="x", markersize=2.5, color=colors[i], linestyle="none",
                label=r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(htmp))

    ax.legend()

    ax.set_xlim(1e0, 1e6)
    ax.set_ylim(1e-10, 1e0)

    ax.set_xlabel(r"Size $s$")
    ax.set_ylabel(r"$p(s)$")

    plt.savefig(args.save)
