import argparse
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()
    
    data = np.load(args.data)
    freqs = data["freqs"] / 1000.
    dts = data["dts"] * 1000.
    rks = data["rks"]
    taus = data["taus"] * 1000.

    colors = cm.viridis(np.linspace(0, 1, len(freqs)))

    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    ax0.set_title(r"(a) AC time constant")
    ax1.set_title(r"(b) AC function")

    ax0.set_yscale("log", nonposy="clip")
    ax0.set_xscale("log", nonposx="clip")

    ax0.plot(freqs, taus[0, :])
    ax0.fill_between(freqs, *taus[1:, :], alpha=0.2)

    for i in range(len(freqs)):
        ax1.plot(dts, rks[0, i, :], color=colors[i])
        ax1.fill_between(dts, *rks[1:, i, :], alpha=0.2, facecolor=colors[i])
    
    ax0.set_xlabel(r"Input rate $\nu$ (Hz)")
    ax0.set_ylabel(r"$\tau$ (\si{\second})")

    ax1.set_xlabel(r"Time lag $\Delta t$ (\si{\second})")
    ax1.set_ylabel(r"$r_{\Delta t}$")

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=len(freqs)),
                           cmap=cm.get_cmap("viridis", len(freqs)))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(len(freqs)) + 0.5)
    cbar.set_ticklabels(np.round(freqs).astype(int))
    cbar.set_label(r"Input rate $\nu$ (Hz)")

    plt.savefig(args.save, transparent=True)
