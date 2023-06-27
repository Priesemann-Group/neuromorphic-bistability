import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    h = data["h"] * 1e-6
    times = data["times"] * 1e3
    freqs = data["freqs"] * 1e-3
    activity = data["activity"] * 1e-3

    colors = [
            [0.71211073, 0.40461361, 0.31372549, 1.],
            [0.23806228, 0.44433679, 0.45444060, 1.]
            ]

    fig = plt.figure(figsize=(2.6, 1.2))
    ax = fig.gca()

    ax.set_xscale("log")

    linestyles = ["dotted", "dashed", "solid"]
    for i, htmp in enumerate(h):
        for t in range(times.size):
            if i == 0:
                line, = ax.plot(freqs, activity[0, i, :, t],
                                label=r"$t = \SI{{{:0.2f}}}{{\second}}".format(times[t]),
                                linestyle=linestyles[t],
                                color=colors[i])
            else:
                line, = ax.plot(freqs, activity[0, i, :, t],
                                linestyle=linestyles[t],
                                color=colors[i])
            ax.fill_between(freqs,
                            activity[0, i, :, t] - activity[1, i, :, t],
                            activity[0, i, :, t] + activity[1, i, :, t],
                            alpha=0.2, facecolor=colors[i])

    style = "Simple, tail_width=0.5, head_width=3, head_length=6"
    kw = dict(arrowstyle=style, color=colors[1])
    p1 = patches.FancyArrowPatch((1e2, 10), (5e1, 72), connectionstyle="arc3,rad=.5", **kw)
    style = "Simple, tail_width=0.5, head_width=2.5, head_length=5"
    kw = dict(arrowstyle=style, color=colors[0])
    p2 = patches.FancyArrowPatch((4e0, 10), (2.5e0, 44), connectionstyle="arc3,rad=.5", **kw)
    style = "Simple, tail_width=0.5, head_width=2.0, head_length=2"
    kw = dict(arrowstyle=style, color=colors[0])
    p2_0 = patches.FancyArrowPatch((3e0, 10), (2.5e0, 36), connectionstyle="arc3,rad=.5", **kw)
    p2_1 = patches.FancyArrowPatch((2.85e0, 32.2), (2.0e0, 39), connectionstyle="arc3,rad=.2", **kw)

    for p in [p1, p2_0, p2_1]:
        plt.gca().add_patch(p)

    ax.text(1e-1, 80, "\hphantom{smooth}onset",
            bbox=dict(boxstyle="square",
                      fc="#CACACA",
                      ec="#CACACA"))

    # leg = ax.legend(loc=2)
    # for handle in leg.legendHandles:
    #     handle.set_color("black")
    ax.set_xlim(freqs[0], 2e2)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 150, 50))

    ax.set_xlabel(r"Stimulus rate $\Delta h$ (\si{\hertz})")
    ax.set_ylabel(r"Rate $\overline{\nu}$ (\si{\hertz})")
    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.12, 0.5)

    plt.savefig(args.save)
