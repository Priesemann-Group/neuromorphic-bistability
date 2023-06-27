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
    h = data["h"] * 1e-6
    taus = data["taus"] * 1e3

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.2, 1.3))
    ax = fig.gca()

    ax.set_xscale("log", nonpositive="mask")
    ax.set_yscale("log", nonpositive="mask")
    for i in range(h.size):
        ax.scatter(taus[i, :, 0], taus[i, :, 1],
                   color=colors[i])
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-3, 1e1)

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="k")

    ax.set_xlabel(r"Fit $\tau_\mathrm{AC}$ (\si{\second})")
    ax.set_ylabel(r"HMM $\tau_\mathrm{HM}$ (\si{\second})")

    plt.savefig(args.save, transparent=True)
