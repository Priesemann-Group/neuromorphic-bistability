import argparse
import numpy as np
import matplotlib.pyplot as plt


def transform(x, C_mem=2.8e-12, tau_mem=20e-6):
    return C_mem * x / tau_mem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    I0_inh_mean = transform(data["I0_inh_mean"]) * 1e6
    I0_inh_std = transform(data["I0_inh_std"]) * 1e6
    I0_inh_scale_mean = transform(data["I0_inh_scale_mean"]) * 1e6
    I0_inh_scale_std = transform(data["I0_inh_scale_std"]) * 1e6

    weights = np.arange(64)

    fig = plt.figure(figsize=(1.5, 1.3))
    ax = fig.gca()

    colors = ["#555555", "#AF5A50"]
    ax.plot(weights, I0_inh_mean, color=colors[0], label="Data")
    ax.fill_between(weights,
                    I0_inh_mean - I0_inh_std,
                    I0_inh_mean + I0_inh_std,
                    alpha=0.2, facecolor=colors[1])
    ax.plot(weights, weights * I0_inh_scale_mean[0] + I0_inh_scale_mean[1],
            color=colors[1], linestyle="--", label="Fit")
    ax.legend(loc=1)

    ax.set_xlabel(r"\vphantom{$\tau_\mathrm{syn}^\mathrm{inh}$}Weight $w_{ij}$")
    ax.set_ylabel(r"\vphantom{$\tau^\mathrm{s,inh}_i$}$\gamma_i w_{ij}^\mathrm{inh}$ (\si{\micro\ampere})")

    ax.set_xlim(0, 63)
    ax.set_ylim([-0.04, 0.0])

    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.3, 0.5)

    plt.savefig(args.save)
