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
    times = data["times"] * 1e3
    boarders = data["boarders"] * 1e3
    dr = data["dr"]

    colors = [
            [0.71211073, 0.40461361, 0.31372549, 1.],
            [0.23806228, 0.44433679, 0.45444060, 1.]
            ]

    fig = plt.figure(figsize=(2.6, 1.2))
    ax = fig.gca()

    for i in range(dr.shape[1]):
        line, = ax.plot(times, dr[0, i, :],
                        label=r"$h = \SI{{{:0.1f}}}{{\kilo\hertz}}$".format(h[i]),
                        color=colors[i])
        ax.fill_between(times,
                        dr[0, i, :] - dr[1, i, :],
                        dr[0, i, :] + dr[1, i, :],
                        alpha=0.2, facecolor=line.get_color())

    for b in boarders:
        ax.axvline(b, linestyle="--", color="#555555")

    ax.axvspan(boarders[0], boarders[0] + 0.2, color="#CACACA")
    ax.axvspan(boarders[1], boarders[1] + 0.2, color="#EAEAEA")

    ax.legend()

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 40)

    ax.set_xlabel(r"Time $t$ (\si{\second})")
    ax.set_ylabel(r"DR $\Delta$")
    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.12, 0.5)

    plt.savefig(args.save)
