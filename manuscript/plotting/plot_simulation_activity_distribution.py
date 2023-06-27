import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    neurons = data["neurons"]
    h = data["h"] * 1e-3
    distribution_bins = data["distribution_bins"]
    distribution = data["distribution"]

    # This can be used to extract the data for the fits
    # data = np.zeros((distribution_bins.size - 1, distribution.shape[0] + 1))
    # data[:, 0] = distribution_bins[:-1]
    # data[:, 1:] = distribution.T
    # np.savetxt("data.txt", data)

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))
    alphas = np.linspace(0.25, 1.0, neurons.size)

    fig = plt.figure(figsize=(1.5, 1.3))
    ax = fig.gca()

    ax.set_yscale("log", nonpositive="clip")

    handles = list()
    for i in np.arange(h.size)[:1]:
        for j in range(neurons.size)[:-1]:
            ax.plot(distribution_bins[:-1] / neurons[j], distribution[i, j, :] * neurons[j],
                    color=colors[i], alpha=alphas[j])

    ax.text(60, 2e-1, r"$N$")
    style = "Simple, tail_width=0.5, head_width=3, head_length=6"
    kw = dict(arrowstyle=style, color="k")
    p = patches.FancyArrowPatch((55, 2e-3), (55, 3e-1), **kw)
    plt.gca().add_patch(p)

    ax.set_xlabel(r"\vphantom{(}Rate $\nu$ (\si{\hertz})")
    ax.set_ylabel(r"Probability $P(\nu)$\hphantom{C}")

    ax.set_ylim(1e-4, 1e0)
    ax.set_xlim(0, 100)

    plt.savefig(args.save, transparent=True)
