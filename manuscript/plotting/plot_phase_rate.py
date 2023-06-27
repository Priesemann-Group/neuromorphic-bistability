import argparse
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    g = data["g"]
    rate = data["rate"] * 1e-3
    h = data["h"][::-1] * 1e-6

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

    for i in range(rate.shape[2]):
        ax.errorbar(g, rate[0, :, rate.shape[2] - (i + 1)],
                    yerr=rate[1, :, rate.shape[2] - (i + 1)],
                    color=colors[i],
                    linestyle="-",
                    fmt="x",
                    ms=2.0,
                    label=r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}".format(h[i]))

    ax.legend(fontsize=6, loc=0)
    ax.axvline(1.0, linewidth=0.8, linestyle="--", color="#555555")

    ax.set_xlabel(r"Ratio $g = w^\mathrm{inh} / w^\mathrm{exc}$")
    ax.set_ylabel(r"Rate $\nu$ (\si{\hertz})")

    ax.yaxis.set_label_coords(-0.32, 0.5)

    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0, 2e2)

    plt.savefig(args.save)
