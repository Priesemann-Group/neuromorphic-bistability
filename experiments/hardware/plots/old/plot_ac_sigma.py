import data
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    conf = data.Sigma()

    data = np.load("data/ac_sigma.npz")
    rks = data["rks"]
    taus = data["taus"]

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

    for i in range(conf.Npaths):
        ax0.plot(conf.sigmas, taus[0, i, :])
        ax0.fill_between(conf.sigmas, *taus[1:, i, :], alpha=0.2)

    ax0.legend(["calibrated", "uncalibrated"])

    for i in range(conf.Nsigmas):
        ax1.plot(dts, rks[0, 0, i, :], color=conf.colors[i])
        ax1.fill_between(dts, *rks[1:, 0, i, :],
                         alpha=0.2, facecolor=conf.colors[i])
    
    for i in range(conf.Nsigmas):
        ax2.plot(dts, rks[0, 1, i, :], color=conf.colors[i])
        ax2.fill_between(dts, *rks[1:, 1, i, :],
                         alpha=0.2, facecolor=conf.colors[i])

    ax0.set_xlabel(r"Noise $\sigma$ (lsb)")
    ax0.set_ylabel(r"$\tau$ (\si{\second})")

    ax1.set_xticklabels([])
    ax1.set_ylabel(r"$r_{\Delta t}$")
    
    ax2.set_xlabel(r"Time lag $\Delta t$ (\si{\second})")
    ax2.set_ylabel(r"$r_{\Delta t}$")

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=conf.Nsigmas),
                           cmap=cm.get_cmap("viridis", conf.Nsigmas))
    sm.set_array([])                                                   

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(conf.Nsigmas) + 0.5)
    cbar.set_ticklabels(conf.sigmas)
    cbar.set_label(r"Noise $\sigma$ (lsb)")

    plt.savefig("plots/ac_sigma.pgf", transparent=True)
    plt.savefig("plots/ac_sigma.pdf", transparent=True)
