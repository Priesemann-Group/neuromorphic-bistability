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

    freqs = data["values0"] / 1000.
    bins = data["bins"] * 1e6
    bins_raster = data["bins_raster"] * 1e6
    boarders = data["boarders"] * 1e6
    
    mi = data["mi"]
    raster_in_cl1 = data["raster_in_cl1"]
    raster_in_cl2 = data["raster_in_cl2"]
    raster_net_cl1 = data["raster_net_cl1"]
    raster_net_cl2 = data["raster_net_cl2"]
    act_in_cl1 = data["act_in_cl1"]
    act_in_cl2 = data["act_in_cl2"]
    act_net_cl1 = data["act_net_cl1"]
    act_net_cl2 = data["act_net_cl2"]

    colors = cm.viridis(np.linspace(0, 1, freqs.size))

    fig = plt.figure(figsize=(3.4, 2.2), constrained_layout=True)
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           width_ratios=[0.48, 0.48, 0.04],
                           height_ratios=[0.35, 0.15, 0.35, 0.15])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[:, 1])
    cax = fig.add_subplot(gs[:, 2])

    ax0.set_title(r"(a) Input")
    ax2.set_title(r"(b) Recurrent")
    ax4.set_title(r"(c) Performance")

    # ax4.set_xscale("log", nonposx="clip")
    ax4.set_yscale("log", nonposy="clip")

    ax0.vlines(raster_in_cl1[:, 0] * 1e6,
               ymin=raster_in_cl1[:, 1], ymax=raster_in_cl1[:, 1] + 1,
               color="#555555")
    ax0.vlines(raster_in_cl2[:, 0] * 1e6,
               ymin=raster_in_cl2[:, 1], ymax=raster_in_cl2[:, 1] + 1,
               color="#AF5A50")

    ax1.plot(bins_raster[:-1], act_in_cl1, color="#555555")
    ax1.plot(bins_raster[:-1], act_in_cl2, color="#AF5A50")
    
    ax2.vlines(raster_net_cl1[:, 0] * 1e6,
               ymin=raster_net_cl1[:, 1], ymax=raster_net_cl1[:, 1] + 1,
               color="#555555")
    ax2.vlines(raster_net_cl2[:, 0] * 1e6,
               ymin=raster_net_cl2[:, 1], ymax=raster_net_cl2[:, 1] + 1,
               color="#AF5A50")

    ax3.plot(bins_raster[:-1], act_net_cl1, color="#555555")
    ax3.plot(bins_raster[:-1], act_net_cl2, color="#AF5A50")
    
    for i in range(mi.shape[1]):
        ax4.plot(bins[:-1], mi[0, i, :], color=colors[i])
        # ax4.fill_between(bins[:-1], *mi[1:, i, :],
        #                  alpha=0.2, facecolor=colors[i])

    for b in boarders:
        ax0.axvline(b, linestyle='--', color="#555555")
        ax1.axvline(b, linestyle='--', color="#555555")
        ax2.axvline(b, linestyle='--', color="#555555")
        ax3.axvline(b, linestyle='--', color="#555555")
        ax4.axvline(b, linestyle='--', color="#555555")

    ax0.set_ylabel(r"Neuron")
    ax1.set_ylabel(r"$a(t)")
    ax2.set_ylabel(r"Neuron")
    ax3.set_ylabel(r"$a(t)")
    ax3.set_xlabel(r"Time $t$ (\si{\milli\second})")
    ax4.set_xlabel(r"Time $t$ (\si{\milli\second})")
    ax4.set_ylabel(r"I (bit)")

    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    ax0.yaxis.set_label_coords(-0.3, 0.5)
    ax1.yaxis.set_label_coords(-0.3, 0.5)
    ax2.yaxis.set_label_coords(-0.3, 0.5)
    ax3.yaxis.set_label_coords(-0.3, 0.5)

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=freqs.size),
                           cmap=cm.get_cmap("viridis", freqs.size))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(freqs.size) + 0.5)
    cbar.set_ticklabels((freqs).astype(int))
    cbar.set_label(r"Input rate $\nu$ (Hz)")

    plt.savefig("plots/letter_delay.pdf", transparent=True)

    # if args.save:
    #     plt.savefig(args.save, transparent=True)
    # else:
    #     plt.show()
