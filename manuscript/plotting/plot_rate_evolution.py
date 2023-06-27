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
    rates = data["rates"] / 1e3
    target = data["target"]
    h = data["h"] * 1e-6
    updates = data["updates"]

    norm = colors.TwoSlopeNorm(vmin=0.58, vmax=1.08, vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.4, 1.3))
    ax = fig.gca()

    for i in range(rates.shape[1]):
        line, = ax.plot(updates, rates[0, i, :],
                        label=r"h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(h[i]),
                        color=colors[i])
        ax.fill_between(updates,
                        rates[0, i, :] - rates[1, i, :],
                        rates[0, i, :] + rates[1, i, :],
                        facecolor=line.get_color(), alpha=0.8)
    ax.axhline(target, linestyle="--", color="#555555")
    ax.legend(loc=4)

    ax.set_xlabel(r"Training iterations")
    ax.set_ylabel(r"Rate $\nu$ (\si{\hertz})")

    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.2, 0.5)

    ax.set_xlim(updates[0], updates[-1])
    ax.set_ylim(0, 15)
    ax.set_xticks(np.arange(0, 1500, 500))

    plt.savefig(args.save)
