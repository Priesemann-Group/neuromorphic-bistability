import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    tau_m = data["tau_m"] * 1e6
    tau_m_uncal = data["tau_m_uncal"] * 1e6

    fig = plt.figure(figsize=(1.5, 1.3))
    ax = fig.gca()

    bins = np.linspace(0, 30, 80)
    ax.hist(tau_m, bins=bins, alpha=0.6)
    ax.hist(tau_m_uncal, bins=bins, alpha=0.6)

    ax.set_xlabel(r"\vphantom{$\tau^\mathrm{s,inh}_i$}$\tau^\mathrm{m}_i$ (\si{\milli\second})")
    ax.set_ylabel(r"Counts")

    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.28, 0.5)

    ax.set_xlim([0, 30])
    ax.set_ylim([0, 200])

    plt.savefig(args.save)
