import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    wexc = data["wexc"]
    winh = data["winh"]
    rates = data["rates"] * 1e-3

    fig = plt.figure(figsize=(0.5, 1.6))
    ax = fig.gca()

    p = ax.pcolormesh(winh, wexc, rates,
                      cmap="Greys",
                      norm=LogNorm(vmin=0.3, vmax=500))

    fig = plt.figure(figsize=(0.1, 1.6))
    ax = fig.gca()
    cbar = matplotlib.colorbar.Colorbar(ax, p, orientation="vertical")
    cbar.set_label(r"Rate $\nu$ (\si{\hertz})")

    plt.savefig(args.save)
