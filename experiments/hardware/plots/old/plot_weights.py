import data
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


if __name__ == "__main__":
    d = np.load("data/weights.npz")
    bins = d["bins"]
    weights_ext = d["weights_ext"]
    weights_int = d["weights_int"]

    conf = data.Standard()
        
    fig = plt.figure(figsize=(3.4, 2.0), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    ax0.set_title(r"(a) Input calib")
    ax1.set_title(r"(b) Recurrent calib")
    
    ax2.set_title(r"(c) Input uncalib")
    ax3.set_title(r"(d) Recurrent uncalib")

    # ax0.set_yscale("log", nonposy="clip")
    # ax1.set_yscale("log", nonposy="clip")
    # ax2.set_yscale("log", nonposy="clip")
    # ax3.set_yscale("log", nonposy="clip")

    for i in range(conf.Nfreqs):
        ax0.plot(bins[:-1], weights_ext[0, 0, i, :], color=conf.colors[i])
        ax1.plot(bins[:-1], weights_int[0, 0, i, :], color=conf.colors[i])
        ax2.plot(bins[:-1], weights_ext[0, 1, i, :], color=conf.colors[i])
        ax3.plot(bins[:-1], weights_int[0, 1, i, :], color=conf.colors[i])

    ax2.set_xlabel(r"Weight (lsb)")
    ax3.set_xlabel(r"Weight (lsb)")

    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    
    ax0.set_ylabel(r"Counts")
    ax2.set_ylabel(r"Counts")

    ax0.set_xlim([0, 63])
    ax1.set_xlim([0, 63])
    ax2.set_xlim([0, 63])
    ax3.set_xlim([0, 63])

    ax0.set_ylim([1e1, 8e1])
    ax1.set_ylim([1e1, 8e1])
    ax2.set_ylim([1e1, 8e1])
    ax3.set_ylim([1e1, 8e1])

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=conf.Nfreqs),
                           cmap=cm.get_cmap("viridis", conf.Nfreqs))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(conf.Nfreqs) + 0.5)
    cbar.set_ticklabels(np.round(conf.freqs / conf.speedup).astype(int))
    cbar.set_label(r"Input rate $\nu$ (Hz)")

    plt.savefig("plots/weights.pgf", transparent=True)
    plt.savefig("plots/weights.pdf", transparent=True)
