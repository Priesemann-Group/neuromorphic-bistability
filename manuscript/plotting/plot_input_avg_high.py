import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    deltah = data["deltah"] * 1e-3
    bins = data["bins"] * 1e3
    boarders = data["boarders"] * 1e3
    activity = data["activity"] * 1e-3

    colors = [
            [0.71211073, 0.40461361, 0.31372549, 1.],
            [0.23806228, 0.44433679, 0.45444060, 1.]
            ]

    fig = plt.figure(figsize=(2.6, 1.2))
    ax = fig.gca()

    for i, h in enumerate(h):
        line, = ax.plot(bins, activity[0, i, :],
                        label=r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(h),
                        color=colors[i])

    for b in boarders:
        ax.axvline(b, linestyle="--", color="#555555")

    ax.axvspan(boarders[0], boarders[0] + 0.2, color="#CACACA")
    ax.axvspan(boarders[1], boarders[1] + 0.2, color="#EAEAEA")

    ax.text(1.475, 50, r"$\Delta h = \SI{{{}}}{{\hertz}}$".format(np.round(deltah)), fontsize=6)
    ax.legend(loc="upper right")

    ax.set_xlabel(r"Time $t$ (\si{\second})")
    ax.set_ylabel(r"Rate $\overline{\nu}$ (\si{\hertz})")
    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.12, 0.5)

    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 80)

    plt.savefig(args.save)
