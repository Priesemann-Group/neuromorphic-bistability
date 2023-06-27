import argparse
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    freqs = data["freqs"]
    bins = data["bins"] * 1000
    raster_low = data["raster_low"]
    raster_high = data["raster_high"]
    activity_low = data["activity_low"]
    activity_high = data["activity_high"]

    fig = plt.figure(figsize=(3.4, 2.0), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.set_title(r"(a) Low input")
    ax1.set_title(r"(b) High input")
    ax2.set_title(r"(c) Low input")
    ax3.set_title(r"(d) High input")

    ax0.vlines(raster_low[:, 0],
               ymin=raster_low[:, 1], ymax=raster_low[:, 1] + 1)
    ax1.vlines(raster_high[:, 0],
               ymin=raster_high[:, 1], ymax=raster_high[:, 1] + 1)
    ax2.plot(bins[:-1] - bins[0], activity_low)
    ax3.plot(bins[:-1] - bins[0], activity_high)

    ax0.set_xticklabels([])
    ax1.set_xticklabels([])

    ax1.set_yticklabels([])
    ax3.set_yticklabels([])

    ax2.set_xlabel(r"Time $t$ (s)")
    ax3.set_xlabel(r"Time $t$ (s)")
    
    ax0.set_ylabel(r"Neuron")
    ax2.set_ylabel(r"$a(t)$")

    plt.savefig(args.save, transparent=True)
