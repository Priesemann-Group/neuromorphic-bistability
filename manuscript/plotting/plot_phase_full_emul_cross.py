import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit


def linear(x, a, b):
    return a * x + b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    wexc = data["wexc"]
    cross = data["cross"]

    norm = colors.TwoSlopeNorm(vmin=0.58, vmax=1.08, vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.gca()
    for i, htmp in enumerate(h):
        # mask = wexc > 7
        # popt, pcov = curve_fit(linear, wexc[mask], cross[i, mask])
        # print(np.round(htmp, 1), popt)

        closest = np.argmin(np.abs(wexc - cross[i, :]))
        ax.plot([wexc[closest], wexc[closest]], [0, wexc[closest]],
                linestyle=":",
                color=colors[i])
        ax.text(closest + 0.4, 0.3, r"$w = {}$".format(closest),
                rotation=90, fontsize=5, color=colors[i])
        ax.plot(wexc, cross[i, :], color=colors[i],
                label=r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(htmp))

    ax.set_xlabel(r"$w^\mathrm{inh}$")
    ax.set_ylabel(r"$w^\mathrm{exc}$")

    ax.plot([wexc[0], wexc[-1]], [wexc[0], wexc[-1]], ls="--", c="k")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)

    plt.savefig(args.save)
