import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    neurons = data["neurons"]
    h = data["h"] * 1e-3
    taus = data["taus"]

    h_ref = [0.58, 0.6, 0.7, 0.8, 0.9, 1.0, 1.08]
    norm = colors.TwoSlopeNorm(vmin=h_ref[0], vmax=h_ref[-1], vcenter=0.9)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    #colors = cmap(norm(h))

    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.4, 1.3))
    ax = fig.gca()

    ax.set_xscale("log")
    ax.set_yscale("log")


    print(taus[0,0,:])
    for i in range(h.size):
        # filter out h=0.7 N=2048 because distribution shows that simulations did no
        # converge properly
        if i==0:
            ax.errorbar(neurons[:-1], taus[0, i, :-1],
                        yerr=taus[1, i, :-1],
                        color=colors[i], fmt="x", ms=3.0,
                        label=r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}".format(h[i]),
                        markeredgewidth=0.8,
                        clip_on=False)
        else:
            ax.errorbar(neurons, taus[0, i, :],
                        yerr=taus[1, i, :],
                        color=colors[i], fmt="x", ms=3.0,
                        label=r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}".format(h[i]),
                        markeredgewidth=0.8,
                        clip_on=False)

    #ax.legend(loc="upper left")
    ax.axhline(20/1000, linestyle="--", color="#555555")
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.1))

    ax.set_xlabel(r"Number of neurons $N$")
    ax.set_ylabel(r"Time $\tau_\mathrm{AC}$ (\si{\second})")

    ax.set_xlim(100, 3000)
    ax.set_ylim(1e-3, 1e1)

    plt.savefig(args.save)
