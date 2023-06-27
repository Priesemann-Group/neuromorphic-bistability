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
    g = data["g"]
    h = data["h"][::-1] * 1e-6
    taus = data["taus"] * 1e3

    # norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.9)
    # cmap = colors.LinearSegmentedColormap.from_list(
    #         "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    # colors = cmap(norm(h))
    norm = colors.TwoSlopeNorm(vmin=h[2], vmax=h[-1], vcenter=0.9)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    c = cmap(norm(h[2:]))
    colors = np.zeros((c.shape[0] + 2, c.shape[1]))
    colors[0, :] = np.array([0.0, 0.0, 0.0, 1.0])
    colors[1, :] = np.array([0.5, 0.5, 0.5, 1.0])
    colors[2:, :] = c

    fig = plt.figure(figsize=(1.4, 1.3))
    ax = fig.gca()

    ax.set_yscale("log")
    for tmp, i in enumerate(range(taus.shape[2])):
        e = ax.errorbar(g, taus[0, :, taus.shape[2] - (i + 1)],
                        yerr=taus[1, :, taus.shape[2] - (i + 1)],
                        color=colors[tmp],
                        linestyle="-",
                        fmt="x",
                        ms=2.0)

        for b in e[2]:
            b.set_clip_on(False)

    ax.set_xlabel(r"Ratio $g = w^\mathrm{inh} / w^\mathrm{exc}$")
    ax.set_ylabel(r"Time $\tau_\mathrm{AC}$ (\si{\second})")

    ax.yaxis.set_label_coords(-0.32, 0.5)

    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.005, 1.0)

    plt.savefig(args.save)
