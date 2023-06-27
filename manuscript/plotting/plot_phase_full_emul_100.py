import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    wexc = data["wexc"]
    winh = data["winh"]
    rates = data["rates"] * 1e-3

    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.gca()
    ax.text(0.95, 0.05,
            r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}".format(h),
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=8,
            transform=ax.transAxes)

    ax.pcolormesh(winh, wexc, rates,
                  cmap="Greys",
                  rasterized=True,
                  norm=LogNorm(vmin=0.3, vmax=500))

    ax.set_xlabel(r"$w^\mathrm{inh}$")
    ax.set_ylabel(r"$w^\mathrm{exc}$")

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    plt.savefig(args.save)
