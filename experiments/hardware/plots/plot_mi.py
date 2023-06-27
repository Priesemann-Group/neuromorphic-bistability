import argparse
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()
    
    data = np.load(args.data)
    delays = data["delays"] * 1000
    freqs = data["freqs"] / 1000.
    mi_ext = data["mi_ext"]
    mi_int = data["mi_int"]

    colors = cm.viridis(np.linspace(0, 1, len(freqs)))

    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    ax0.set_title(r"(a) Input - Neuron")
    ax1.set_title(r"(b) Neuron - Neuron")

    ax0.set_xscale("log", nonposx="clip")
    ax0.set_yscale("log", nonposy="clip")
    ax1.set_xscale("log", nonposx="clip")
    ax1.set_yscale("log", nonposy="clip")

    for i in range(len(freqs)):
        ax0.plot(delays, mi_ext[0, i, :], color=colors[i])
        ax1.plot(delays, mi_int[0, i, :], color=colors[i])

    ax0.set_xlabel(r"Delay $\tau$ (s)")
    ax1.set_xlabel(r"Delay $\tau$ (s)")
    
    ax0.set_ylabel(r"MI (bit)")

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=freqs.size),
                           cmap=cm.get_cmap("viridis", freqs.size))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(freqs.size) + 0.5)
    cbar.set_ticklabels(np.round(freqs).astype(int))
    cbar.set_label(r"Input rate $\nu$ (Hz)")

    if args.save:
        plt.savefig(args.save, transparent=True)
    else:
        plt.show()
