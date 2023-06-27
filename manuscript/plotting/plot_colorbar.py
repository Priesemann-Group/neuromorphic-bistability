import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    h = np.load(args.data)["h"] * 1e-6
    h = np.round(h, 2)
    # h = h[1:][::2]

    # fig = plt.figure(figsize=(0.1, 1.3))
    fig = plt.figure(figsize=(8.43, 0.15))
    cax = fig.gca()

    norm = colors.TwoSlopeNorm(vmin=h[0], vmax=h[-1], vcenter=0.8)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])

    center = np.where(h == 0.8)[0]
    norm = colors.TwoSlopeNorm(vmin=0, vmax=h.size, vcenter=h.size // 2)
    sm = cm.ScalarMappable(norm=norm,
                           cmap=cmap)
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="horizontal")
    # cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks(np.arange(h.size) + 0.5)
    cbar.set_ticklabels(["{:0.2f}".format(htmp) for htmp in h])
    cbar.set_label(r"Input rate $h$ (\si{\kilo\hertz})")
    # cax.yaxis.set_ticks_position("left")
    # cax.yaxis.set_label_position("left")

    plt.savefig(args.save, transparent=True, dpi=600)
