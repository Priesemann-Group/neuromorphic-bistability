import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    v_thres = data["v_thres"]
    v_thres_uncal = data["v_thres_uncal"]

    fig = plt.figure(figsize=(1.5, 1.3))
    ax = fig.gca()

    bins = np.linspace(0.2, 0.8, 80)
    ax.hist(v_thres, bins=bins, alpha=0.6)
    ax.hist(v_thres_uncal, bins=bins, alpha=0.6)

    ax.set_xlabel(r"$u^\mathrm{thres}_i$ (\si{\volt})")
    ax.set_ylabel(r"Counts")

    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.28, 0.5)

    ax.set_xlim([0.2, 0.8])
    ax.set_ylim([0, 200])
    ax.set_xticks(np.arange(0.2, 0.9, 0.2))

    plt.savefig(args.save)
