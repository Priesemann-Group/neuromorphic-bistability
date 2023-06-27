import data
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    conf = data.Standard()

    data = np.load("data/ac.npz")
    rks = data["rks"]
    taus = data["taus"]

    colors = cm.viridis(np.linspace(0, 1, conf.Nfreqs))
    dts = conf.steps * conf.binwidth * conf.speedup

    fig = plt.figure(figsize=(3.4, 2.0), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    ax0.set_title(r"(a) AC time constant")
    ax1.set_title(r"(b) AC calibrated")
    ax2.set_title(r"(c) AC uncalibrated")

    ax0.set_yscale("log", nonposy="clip")
    ax0.set_xscale("log", nonposx="clip")

    for i in range(conf.Npaths):
        ax0.plot(conf.freqs / 1000., taus[0, i, :])
        ax0.fill_between(conf.freqs / 1000., *taus[1:, i, :], alpha=0.2)

    ax0.legend(["calibrated", "uncalibrated"])

    for i in range(conf.Nfreqs):
        ax1.plot(dts, rks[0, 0, i, :], color=colors[i])
        ax1.fill_between(dts, *rks[1:, 0, i, :], alpha=0.2, facecolor=colors[i])
    
    for i in range(conf.Nfreqs):
        ax2.plot(dts, rks[0, 1, i, :], color=colors[i])
        ax2.fill_between(dts, *rks[1:, 1, i, :], alpha=0.2, facecolor=colors[i])

    ax0.set_xlabel(r"Input rate $\nu$ (Hz)")
    ax0.set_ylabel(r"$\tau$ (\si{\second})")

    ax1.set_xticklabels([])
    ax1.set_ylabel(r"$r_{\Delta t}$")
    
    ax2.set_xlabel(r"Time lag $\Delta t$ (\si{\second})")
    ax2.set_ylabel(r"$r_{\Delta t}$")

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=conf.Nfreqs),
                           cmap=cm.get_cmap("viridis", conf.Nfreqs))
    sm.set_array([])                                                   

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(conf.Nfreqs) + 0.5)
    cbar.set_ticklabels((conf.freqs / float(conf.speedup)).astype(int))
    cbar.set_label(r"Input rate $\nu$ (Hz)")

    plt.savefig("plots/ac.pgf", transparent=True)
    plt.savefig("plots/ac.pdf", transparent=True)
