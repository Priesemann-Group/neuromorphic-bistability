import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    w_exc = data["w_exc"]
    w_inh = data["w_inh"]
    spikes = data["spikes"]
    bins = data["bins"]
    activity = data["activity"] * 1e-3

    fig = plt.figure(figsize=(1.4, 2.0))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.7, 0.3])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.vlines(spikes[:, 0] * 1e3,
               ymin=spikes[:, 1], ymax=spikes[:, 1] + 1,
               color="#AF5A50",
               rasterized=True)
    ax1.plot(bins * 1e3, activity, color="#AF5A50")

    ax0.text(0.95, 0.95,
             r"$g={}/{}$".format(w_inh, w_exc),
             verticalalignment="top",
             horizontalalignment="right",
             fontsize=8,
             transform=ax0.transAxes)

    ax0.set_xticklabels([])

    ax1.set_xlabel(r"Time $t$ (\si{\second})")
    ax0.set_ylabel(r"\vphantom{(\si{\hertz})}Neuron")
    ax1.set_ylabel(r"Rate $\nu$ (\si{\hertz})")

    ax0.yaxis.set_label_coords(-0.32, 0.5)
    ax1.yaxis.set_label_coords(-0.32, 0.5)

    ax0.set_xlim(0, 3)
    ax1.set_xlim(0, 3)
    ax0.set_ylim(0, 512)
    ax1.set_ylim(0, 150)

    ax1.set_yticks(np.arange(0, 200, 50))

    plt.savefig(args.save)
