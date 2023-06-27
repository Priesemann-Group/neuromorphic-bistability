import os
import argparse
import numpy as np
import multiprocessing as mp
from functools import partial
from hmmlearn import hmm

from helpers import average_ci
from helpers import shape_iter

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def get_std(indices, args):
    try:
        data = np.load(args.path.format(
            args.neurons[indices[2]], args.Kext[indices[0]], args.seeds[indices[1]]))
    except FileNotFoundError:
        print("MISSING", indices)
        return np.array([np.nan, np.nan])
    spikes = data['spikes']
    time = data['time']
    bins = np.arange(time[0], time[-1] + time[1] - time[0], args.binwidth)
    # Calculate hmm model on activity because hmmlearn needs integers
    activity = np.histogram(spikes[:, 0] / 1000, bins=bins)[0]

    remodel = hmm.MultinomialHMM(n_components=2, n_iter=100)
    remodel.fit(activity.reshape((activity.size, -1)))
    prediction = remodel.predict(activity.reshape((activity.size, -1))).astype(bool)

    activity_0 = activity[~prediction].mean()
    activity_1 = activity[prediction].mean()
    if np.argmax([activity_0, activity_1]) == 0:
        prediction = ~prediction

    activity_up = activity[prediction] / args.neurons[indices[2]] / args.binwidth
    activity_down = activity[~prediction] / args.neurons[indices[2]] / args.binwidth
    std_up = np.std(activity_up)
    std_down = np.std(activity_down)
    return np.array([std_up, std_down])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--saveto", type=str, default="./plots/")
    parser.add_argument("--neurons", type=int, default=[512], nargs="+")
    parser.add_argument("--seeds", type=int, nargs='+', default=np.arange(10000, 20000, 1000))
    parser.add_argument("--Kext", type=int, nargs='+', default=[66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110])
    parser.add_argument("--binwidth", type=float, default=5e-3)
    args = parser.parse_args()
    print(args.Kext)
    print(args.seeds)

    filename = 'data_brian_N{}_g1.0_Kext{}_seed{}.npz'
    args.path = os.path.join(args.path, filename)
    shape = (len(args.Kext), len(args.seeds), len(args.neurons))

    with mp.Pool() as pool:
        stds = np.array(pool.map(partial(get_std, args=args),
                                 shape_iter(shape)))
    stds = stds.reshape(shape + (2, ))
    stds = average_ci(stds)

    stds[np.isnan(stds)] = 0

    #
    # PLOT
    #
    fig = plt.figure(figsize=(3.4, 1.6), constrained_layout=True)
    ax = fig.gca()

    # ax.set_yscale("log", nonpositive="clip")
    handles = []
    # linestyles = ["-", ":", "--", "-."]
    linestyles = ["-", "-", "-", "-"]
    alphas = [1, 0.75, 0.5, 0.25]
    for j, N in enumerate(args.neurons):
        ax.plot(args.Kext, stds[0, :, j, 0], linestyle=linestyles[j], color="#AF5A50", alpha=alphas[j])
        ax.plot(args.Kext, stds[0, :, j, 1], linestyle=linestyles[j], color="#aaaaaa", alpha=alphas[j])
        ax.fill_between(args.Kext, *stds[1:, :, j, 0], color="#AF5A50", alpha=0.2)
        ax.fill_between(args.Kext, *stds[1:, :, j, 1], color="#aaaaaa", alpha=0.2)

        handles.append(mlines.Line2D([], [], color='gray', label=args.neurons[j], alpha=alphas[j],
                       linestyle=linestyles[j]))

    handles.append(mlines.Line2D([], [], color='gray', label="down"))
    handles.append(mlines.Line2D([], [], color='#AF5A50', label="up"))

    ax.legend(loc="lower right", handles=handles)

    ax.set_xlabel(r"average indegree")
    ax.set_ylabel(r"$\sigma_\nu$ (Hz)")
    ax.set_ylim(0, None)

    plt.savefig(f"{args.saveto}figure_N{args.neurons}_variance.pdf", transparent=True)
