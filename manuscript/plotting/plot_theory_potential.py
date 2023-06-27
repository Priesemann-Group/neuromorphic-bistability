import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    # specify parameters
    sigma=50
    N=512
    tau=10
    alpha=30
    beta=15
    b=25

    sigma2=sigma*sigma

    # specify things to plot
    rhos = np.arange(0.01,1,0.001)
    #hs = np.array([0.1,0.4,0.8])
    hs=np.array([0.1,0.4,0.7])

    def V(rho, h, tau, alpha, beta, b, sigma2, N):
        return (sigma2/2/N-h)*np.log(rho) + (tau-alpha+h*(1+beta))*rho + b/2.*np.power(rho,2.)

    norm = colors.TwoSlopeNorm(vmin=hs[0], vmax=hs[-1], vcenter=0.5)
    cmap = colors.LinearSegmentedColormap.from_list(
            "my", [(0, "#AF5A50"), (0.5, "#D7AA50"), (1, "#005B82")])
    colors = cmap(norm(hs))

    fig = plt.figure(figsize=(1.5, 1.3))
    ax = fig.gca()

    for i in np.arange(hs.size-1,-1,-1):
        ax.plot(rhos, V(rhos,hs[i],tau, alpha, beta, b, sigma2, N)-V(0.01,hs[i],tau, alpha, beta, b, sigma2, N),
                color=colors[i],
                label=r"h = {}".format(hs[i])
                )

    ax.legend(loc="lower right")
    ax.set_xlabel(r"Fraction $\rho$")
    ax.set_ylabel(r"Potential $V(\rho)$")

    ax.set_ylim(0, 6)
    ax.xaxis.set_label_coords(0.5, -0.18)

    plt.savefig(args.save, transparent=True)
    #plt.show()
