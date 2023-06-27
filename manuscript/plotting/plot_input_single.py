import argparse
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    bins = data["bins"] * 1e3
    boarders = data["boarders"] * 1e3
    activity = data["activity"] * 1e-3

    colors = [
            [0.71211073, 0.40461361, 0.31372549, 1.],
            [0.23806228, 0.44433679, 0.45444060, 1.]
            ]

    fig = plt.figure(figsize=(2.6, 1.2))
    gs = gridspec.GridSpec(h.size, 1, figure=fig, hspace=0.35)

    ax = [fig.add_subplot(g) for g in gs]
    fax = fig.add_subplot(gs[:])

    linestyles = ["solid", "dotted", "dashed"]
    alphas = np.array([0.2, 0.2, 1.0])
    for i, kin in enumerate(activity):
        for j, seed in enumerate(kin):
            ax[i].plot(bins, seed, color=colors[i],
                       alpha=alphas[j])

        ax[i].text(1.5, 60, r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}".format(h[i]),
                   fontsize=6)

        ax[i].set_xlim(bins[0], bins[-1])
        ax[i].set_ylim(0, 80)
        ax[i].set_yticks(np.arange(0, 100, 40))

        for b in boarders:
            ax[i].axvline(b, linestyle="--", color="#555555")

        ax[i].axvspan(boarders[0], boarders[0] + 0.2, color="#CACACA")
        ax[i].axvspan(boarders[1], boarders[1] + 0.2, color="#EAEAEA")

        if i < (h.size - 1):
            ax[i].set_xticklabels([])

    fax.spines["right"].set_visible(False)
    fax.spines["left"].set_visible(False)
    fax.spines["bottom"].set_visible(False)
    fax.spines["top"].set_visible(False)

    fax.set_xticks([])
    fax.set_yticks([])
    fax.set_xlabel(r"Time $t$ (\si{\second})")
    fax.set_ylabel(r"Rate $\nu$ (\si{\hertz})")
    fax.xaxis.set_label_coords(0.5, -0.18)
    fax.yaxis.set_label_coords(-0.12, 0.5)

    plt.savefig(args.save, transparent=True)
