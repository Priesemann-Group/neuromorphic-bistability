import argparse
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    d = np.load(args.data)
    bins = d["bins"]
    freqs = d["freqs"] / 1000.
    weights_ext = d["weights_ext"]
    weights_int = d["weights_int"]

    colors = cm.viridis(np.linspace(0, 1, freqs.size))
        
    fig = plt.figure(figsize=(3.4, 1.2), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    ax0.set_title(r"(a) Input")
    ax1.set_title(r"(b) Recurrent")
    
    ax0.set_yscale("log", nonposy="clip")
    ax1.set_yscale("log", nonposy="clip")

    for i in range(freqs.size):
        ax0.plot(bins[:-1], weights_ext[0, i, :], color=colors[i])
        ax1.plot(bins[:-1], weights_int[0, i, :], color=colors[i])

    ax0.set_xlabel(r"Weight (lsb)")
    ax1.set_xlabel(r"Weight (lsb)")

    ax1.set_yticklabels([])
    
    ax0.set_ylabel(r"Counts")

    ax0.set_ylim([1e1, 4e2])
    ax1.set_ylim([1e1, 4e2])

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=freqs.size),
                           cmap=cm.get_cmap("viridis", freqs.size))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(freqs.size) + 0.5)
    cbar.set_ticklabels(np.round(freqs).astype(int))
    cbar.set_label(r"Input rate $\nu$ (Hz)")

    plt.savefig(args.save, transparent=True)
