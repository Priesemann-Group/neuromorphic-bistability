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
    perts = data["pert"]

    colors = cm.viridis(np.linspace(0, 1, len(freqs)))

    fig = plt.figure(figsize=(2.4, 1.6), constrained_layout=True)
    ax = fig.gca()

    ax.set_title(r"Susceptibility")

    ax.set_xscale("log", nonposx="clip")

    ax.plot(freqs, perts[0, :])
    ax.fill_between(freqs, *perts[1:, :], alpha=0.2)

    ax.set_xlabel(r"Input rate $\nu$ (Hz)")
    ax.set_ylabel(r"$\chi$")

    plt.savefig(args.save, transparent=True)
