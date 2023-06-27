import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()
    
    data = np.load(args.data)
    freqs = data["freqs"] / 1000.
    rates = data["rates"] / 1000.
    target = data["target"]

    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)

    ax = fig.gca()

    ax.set_xscale("log", nonposx="clip")

    ax.plot(freqs, rates[0, :])
    ax.fill_between(freqs, *rates[1:, :], alpha=0.2)

    ax.axhline(target, linestyle="--", alpha=0.5)

    ax.set_ylim([0, 2.0 * target])

    ax.set_xlabel(r"Input rate $\nu$ (Hz)")
    ax.set_ylabel(r"Firing rate (Hz)")

    plt.savefig(args.save, transparent=True)
