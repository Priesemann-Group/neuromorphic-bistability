import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    data = np.load(args.data)
    ps = data["ps"]
    lams = data["lams"] / 2. * 5
    std = data["std"] * 1e-3
    lam_select = data["lam_select"]
    ps_select = data["ps_select"]

    fig = plt.figure(figsize=(3.5, 1.3))
    ax = fig.gca()

    im = ax.imshow(std.T, origin="lower", cmap="Greys", vmin=0, vmax=10)
    ax.scatter(lam_select, ps_select,
               marker="*", s=20, color="r")
    ax.set_xlabel(r"Learning rate $\lambda$ (\si{\per\milli\second})")
    ax.set_ylabel(r"Probability $p$ (\si{\percent})")

    ax.set_yticks(np.arange(ps.size))
    ax.set_xticks(np.arange(lams.size))
    ax.set_yticklabels(["{:1.1f}".format(p) for p in ps])
    ax.set_xticklabels(["{:1.2f}".format(lam) for lam in lams], rotation=50, ha="center")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\sqrt{\left\langle \left(\nu_i - \nu^\ast\right)^2\right\rangle}$")

    plt.savefig(args.save)
