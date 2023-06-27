import os
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import partial
from scipy.optimize import curve_fit

from helpers import average_se
from helpers import shape_iter


def func(x, tau):
    return np.exp(-x / tau)


def coefficients(indices, args):
    rk = np.full(args.steps.size, np.nan)
    try:
        path = args.path.format(
            args.neurons[indices[2]], args.Kext[indices[0]], args.seeds[indices[1]])
        print(path)
        data = np.load(args.path.format(
            args.neurons[indices[2]], args.Kext[indices[0]], args.seeds[indices[1]]))
    except FileNotFoundError:
        print("MISSING", indices)
        return rk
    spikes = data['spikes']
    time = data['time']
    bins = np.arange(time[0], time[-1] + time[1] - time[0], args.binwidth)
    activity = np.histogram(spikes[:, 0] / 1000, bins=bins)[0]
    for i, step in enumerate(args.steps):
        front = activity[:-step] - activity[:-step].mean()
        back = activity[step:] - activity[step:].mean()
        rk[i] = np.mean(front * back) / activity[:-step].var()
    return rk


def fit(rk, steps):
    tau0 = 50
    a0 = rk[0] / np.exp(-steps[0] / tau0)
    o0 = rk[-1]
    p0 = np.array([a0, tau0, o0])
    try:
        res = curve_fit(func, steps, rk, p0=50.)
        return res[0][0]
        # if res[0][0] < 0.1 or abs(res[0][2]) > 0.2:
        #     print("failed", "A, tau, o", res[0])
        #     return np.nan
        # else:
        #     return res[0][1]
    except Exception as e:
        print("ignoring", repr(e))
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--saveto", type=str, default="./plots/")
    parser.add_argument("--neurons", type=int, default=[256, 512, 786, 1024], nargs="+")
    parser.add_argument("--seeds", type=int, nargs='+', default=np.arange(567800, 567850, 1))
    parser.add_argument("--Kext", type=int, nargs='+', default=np.array([70, 90, 110]))
    parser.add_argument("--binwidth", type=float, default=5e-3)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    print(args.Kext)
    print(args.seeds)

    args.steps = np.arange(1, args.steps)

    filename = 'data_brian_N{}_g1.0_Kext{}_seed{}.npz'
    args.path = os.path.join(args.path, filename)
    shape = (len(args.Kext), len(args.seeds), len(args.neurons))

    with mp.Pool() as pool:
        rks = np.array(pool.map(partial(coefficients, args=args),
                                shape_iter(shape)))
        taus = np.array(pool.map(partial(fit, steps=args.steps), rks))

    rks = rks.reshape(shape + (args.steps.size, ))
    rks = average_se(rks)

    taus = taus.reshape(shape)
    taus = average_se(taus)
    taus *= args.binwidth
    dts = args.steps * args.binwidth
    taus[np.isnan(taus)] = 0
    print(taus[0])

    np.savez("simulation_ac.npz", taus=taus, neurons=args.neurons, h=np.array(args.Kext)*10.0)
    # np.savez("si_ac.npz", taus=taus, rks=rks, dts=dts, h=np.array(args.Kext)*10.0)

    #
    # PLOT
    #
    colors = cm.viridis(np.linspace(0, 1, len(args.Kext)))

    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.48, 0.48, 0.04])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    ax0.set_title(r"(a) AC time constant")
    ax1.set_title(r"(b) AC function")

    ax0.set_yscale("log", nonpositive="clip")
    # ax0.set_xscale("log", nonpositive="clip")

    # select num neurons to be plotted
    j = 0

    ax0.plot(args.Kext, taus[0, :, j])
    ax0.fill_between(args.Kext, *taus[1:, :, j], alpha=0.2)

    for i in range(len(args.Kext)):
        ax1.plot(dts, rks[0, i, j, :], color=colors[i])
        ax1.fill_between(dts, *rks[1:, i, j, :], alpha=0.2, facecolor=colors[i])

    ax0.set_xlabel(r"average indegree")
    ax0.set_ylabel(r"$\tau$ (\si{\second})")

    ax1.set_xlabel(r"Time lag $\Delta t$ (\si{\second})")
    ax1.set_ylabel(r"$r_{\Delta t}$")

    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=len(args.Kext)),
                           cmap=cm.get_cmap("viridis", len(args.Kext)))
    sm.set_array([])

    cbar = matplotlib.colorbar.Colorbar(cax, sm, orientation="vertical")
    cbar.set_ticks(np.arange(len(args.Kext)) + 0.5)
    cbar.set_ticklabels(np.round(args.Kext, 1))
    cbar.set_label(r"average indegree")

    plt.savefig(f"{args.saveto}figure_N{args.neurons[j]}_ac.pdf", transparent=True)

    # plot tau(N)
    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    ax = fig.gca()
    ax.set_yscale("log", nonpositive="clip")
    # ax0.set_xscale("log", nonpositive="clip")
    # linestyles = ["-", ":", "--", "-."]
    linestyles = ["-", "-", "-", "-"]
    alphas = [1, 0.75, 0.5, 0.25]
    for j, N in enumerate(args.neurons):
        ax.plot(args.Kext, taus[0, :, j], linestyle=linestyles[j], color="grey", label=N, alpha=alphas[j])
        ax.fill_between(args.Kext, *taus[1:, :, j], alpha=0.2, color="grey")

    ax.set_xlabel(r"average indegree")
    ax.set_ylabel(r"$\tau$ (\si{\second})")

    fig.legend()
    plt.savefig(f"{args.saveto}figure_N{args.neurons}_ac_compare.pdf", transparent=True)
