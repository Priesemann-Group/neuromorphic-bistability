import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    d = np.load(args.data)
    bins = d["bins"]
    h = d["h"] * 1e-3
    h_selection = np.array([0.6, 0.8, 1.0])
    weights = d["weights"]

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(3.55, 1.3))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1)
    ax = [fig.add_subplot(g) for g in gs]
    fax = fig.add_subplot(gs[:])

    for i, htmp in enumerate(h_selection):
        idx = np.where(htmp == h)[0][0]
        line, = ax[0].plot(-bins[:-1], weights[0, i, 1, :],
                           color=colors[idx])

        ax[0].fill_between(-bins[:-1], 
                           weights[0, i, 1, :] - weights[1, i, 1, :],
                           weights[0, i, 1, :] + weights[1, i, 1, :],
                           alpha=0.2, facecolor=line.get_color())
        line, = ax[1].plot(bins[:-1], weights[0, i, 0, :],
                           color=colors[idx])

        ax[1].fill_between(bins[:-1],
                           weights[0, i, 0, :] - weights[1, i, 0, :],
                           weights[0, i, 0, :] + weights[1, i, 0, :],
                           alpha=0.2, facecolor=line.get_color())

    ax[0].set_ylim([1e-1, 1e4])
    ax[1].set_ylim([1e-1, 1e4])
    ax[0].set_yscale("log", nonpositive="mask")
    ax[1].set_yscale("log", nonpositive="mask")

    ax[0].text(-32, 4e3, "Inhibitory", fontsize=7, ha="center")
    ax[1].text(32, 4e3, "Excitatory", fontsize=7, ha="center")
    ax[0].set_xlim([-63, 0])
    ax[1].set_xlim([0, 63])
    ax[0].set_xticks(np.arange(-64, 1, 32))
    ax[1].set_xticks(np.arange(0, 65, 32))
    ax[0].set_yticks([1e-1, 1e1, 1e3, 1e5])
    ax[1].set_yticklabels([])

    fax.spines["right"].set_visible(False)
    fax.spines["left"].set_visible(False)
    fax.spines["bottom"].set_visible(False)
    fax.spines["top"].set_visible(False)

    fax.set_xticks([])
    fax.set_yticks([])
    fax.set_ylabel(r"Counts")
    fax.set_xlabel(r"\vphantom{(\si{\percent})}Signed Weight $c_{ij}w_{ij}^\mathrm{rec}$")
    fax.xaxis.set_label_coords(0.5, -0.18)
    fax.yaxis.set_label_coords(-0.09, 0.5)

    plt.savefig(args.save, transparent=True)
