import argparse
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
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
    h_selection = data["h_selection"] * 1e-6
    weights = data["weights"]

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.4, 1.3))
    ax = fig.gca()

    marker = ["^", "o"]
    labels = ["Excitatory", "Inhibitory"]
    for j in range(weights.shape[2]):
        # popt, pcov = curve_fit(linear, h, weights[0, :, j, 0], sigma=weights[1, :, j, 0], absolute_sigma=True)

        # r = weights[0, :, j, 0] - linear(h, *popt)
        # chisq = np.sum((r / weights[1, :, j, 0]) ** 2)
        # print(labels[j], popt, np.sqrt(pcov), chisq, chisq / float(h.size - 2))
        for i in range(weights.shape[1]):
            if np.any(h[i] == h_selection):
                color = colors[i]
                fill_style = "full"
            else:
                color = "#656565"
                fill_style = "none"
            if i == 0:
                ax.errorbar(h[i], weights[0, i, j, 0],
                            yerr=weights[1, i, j, 0],
                            color=color, fmt=marker[j], ms=3.5, label=labels[j],
                            linewidth=0.8, markeredgewidth=0.6, fillstyle=fill_style)
            else:
                ax.errorbar(h[i], weights[0, i, j, 0],
                            yerr=weights[1, i, j, 0],
                            color=color, fmt=marker[j], ms=3.5, linewidth=0.6,
                            markeredgewidth=0.6, fillstyle=fill_style)

    ax.legend(loc="lower left")
    ax.set_xlim(0.55, 1.1)
    ax.set_ylim(6, 16)

    ax.set_yticks(np.arange(6, 18, 2))

    ax.set_xlabel(r"Input rate $h$ (\si{\kilo\hertz})")
    ax.set_ylabel(r"Mean weight $w^\mathrm{rec}$\hphantom{C}")

    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.25, 0.5)

    plt.savefig(args.save)
