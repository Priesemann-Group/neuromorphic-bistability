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
    bins = data["bins"] * 1e6
    freqs = data["freqs"] / 1000.
    perts = data["pert"]

    baseline = np.mean(perts[0, :, :], axis=1)

    colors = cm.viridis(np.linspace(0, 1, len(freqs)))

    fig = plt.figure(figsize=(2.4, 1.6), constrained_layout=True)
    ax = fig.gca()

    ax.set_title(r"Susceptibility")

    ax.axvline(0, linestyle='--')
    for i in range(perts.shape[1]):
        ax.plot(bins[:-1], perts[0, i, :] - baseline[i], color=colors[i])
        ax.fill_between(bins[:-1], *perts[1:, i, :] - baseline[i], color=colors[i], alpha=0.2)

#    ax.set_xlim([-50, 50])

    ax.set_xlabel(r"Time to perturbation $t$ (\si{\milli\second})")
    ax.set_ylabel(r"$a(t) / \bar{a}(t)$")

    plt.savefig(args.save, transparent=True)
