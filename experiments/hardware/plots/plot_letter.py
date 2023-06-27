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
    
    data = np.load(args.data)

    mi = data["mi"]
    freqs = data["values0"] / 1000.
    letter = np.arange(mi.shape[2]) + 1
        
    colors = cm.viridis(np.linspace(0, 1, freqs.size))

    fig = plt.figure(figsize=(2.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.95, 0.05])

    ax0 = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    ax0.set_yscale("log", nonposy="clip")

    for i in range(mi.shape[1]):
        ax0.plot(letter, mi[0, i, :], color=colors[i])
        ax0.fill_between(letter, *mi[1:, i, :], alpha=0.2, facecolor=colors[i])

    ax0.set_xlabel(r"Letter")
    ax0.set_ylabel(r"I (bit)")

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
