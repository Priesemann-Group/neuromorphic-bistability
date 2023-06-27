import argparse
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()
    
    data = np.load(args.data)

    vrd = data["vrd"]
    decay = data["decay"] * 1e6
    freqs = data["values0"] / 1000.
    bins = data["bins"] * 1e6
    boarders = data["boarders"] * 1e6

    colors = cm.viridis(np.linspace(0, 1, freqs.size))
    linestyles = ["-", ":", "--"]

    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    ax0.set_title(r"(a) Temporal")
    ax1.set_title(r"(b) Integrated")
    ax2.set_title(r"(c) Decay")

    ax1.set_xscale("log", nonposx="clip")

    ax0.set_yscale("log", nonposy="clip")
    ax1.set_yscale("log", nonposy="clip")
    # ax2.set_yscale("log", nonposy="clip")

    for i in range(vrd.shape[1]):
        for j in range(vrd.shape[2]):
            ax0.plot(bins - bins[0], vrd[0, i, j, :], color=colors[i],
                     linestyle=linestyles[j])
            ax0.fill_between(bins - bins[0], *vrd[1:, i, j, :],
                             alpha=0.2, facecolor=colors[i])

    for b in boarders:
        ax0.axvline(b - bins[0], linestyle='--', color="#555555", linewidth=0.5)
        ax2.axvline(b - bins[0], linestyle='--', color="#555555", linewidth=0.5)

    lines = [Line2D([0], [0], color="#555555", linestyle=l) for l in linestyles]
    ax0.legend(lines, ["T2T", "Class"], handlelength=0.6)

    ax1.plot(freqs, vrd[0, :, 0, 0])
    ax1.fill_between(freqs, *vrd[1:, :, 0, 0], alpha=0.2)

    for i in range(vrd.shape[1]):
        diff = (vrd[:, i, 1, :] - vrd[:, i, 0, :])
        diff -= diff[0, :5].mean()
        ax2.plot(bins - bins[0], diff[0, :] / diff[0, :].max(),
                 color=colors[i])
        ax2.fill_between(bins - bins[0], *(diff[1:, :] / diff[0, :].max()),
                         alpha=0.2, facecolor=colors[i])
    # ax2.plot(freqs, decay)
    # ax2.fill_between(freqs, *decay[1:, :], alpha=0.2)

    ax0.set_xlabel(r"Time (\si{\milli\second})")
    ax2.set_xlabel(r"$\nu$ (\si{\hertz})")
    ax0.set_ylabel(r"vRD")
    ax1.set_ylabel(r"vRD")
    ax2.set_ylabel(r"$\tau_\mathrm{vRD}$ (\si{\milli\second})")

    ax1.set_xlim([0.5, 30])

    ax0.set_ylim([9e3, 1.1e5])
    ax1.set_ylim([9e3, 1.5e5])

    ax1.set_xticklabels([])

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=freqs.size),
                           cmap=cm.get_cmap("viridis", freqs.size))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(freqs.size) + 0.5)
    cbar.set_ticklabels((freqs).astype(int))
    cbar.set_label(r"Input rate $\nu$ (Hz)")

    if args.save:
        plt.savefig(args.save, transparent=True)
    else:
        plt.show()
