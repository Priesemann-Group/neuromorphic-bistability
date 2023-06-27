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
    h = data["h"] * 1e-3
    rate = data["rate"]
    target = 10.
    h_selection = np.array([0.6, 0.8, 1.0])

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(h))

    fig = plt.figure(figsize=(1.4, 1.3))
    ax = fig.gca()

    ax.axhline(target, linestyle="--", color="#555555")
    for i in range(rate.shape[1]):
        if np.any(h[i] == h_selection):
            color = colors[i]
        else:
            color = "#656565"
        e = ax.errorbar(h[i], rate[0, i],
                    yerr=rate[1, i],
                    color=color, fmt="x", ms=3.0,
                    linewidth=0.8, markeredgewidth=0.6)
        for b in e[2]:
            b.set_clip_on(False)

    ax.set_xlabel(r"Input rate $h$ (\si{\kilo\hertz})")
    ax.set_ylabel(r"Rate $\nu$ (\si{\hertz})")

    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.2, 0.5)

    ax.set_xlim(0.55, 1.1)
    ax.set_ylim(8, 20)

    plt.savefig(args.save, transparent=True)
