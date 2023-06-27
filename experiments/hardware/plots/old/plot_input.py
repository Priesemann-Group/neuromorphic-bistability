import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    in_spikes = np.load("../experiments/test_letters_in.npy")
    in_spikes_other = np.load("../experiments/test_letters_in_other.npy")

    fig = plt.figure(figsize=(2.4, 1.4), constrained_layout=True)
    ax = fig.gca()
    ax.grid(False)

    for i in range(6):
        ax.axvline(20e-6*i*1000, color="#555555", linewidth=0.5, linestyle="--")
    ax.vlines(in_spikes[:, 0] * 1000,
              ymin=in_spikes[:, 1], ymax=in_spikes[:, 1] + 1,
              color="#AF5A50")
    ax.vlines(in_spikes_other[:, 0] * 1000,
              ymin=in_spikes_other[:, 1], ymax=in_spikes_other[:, 1] + 1,
              color="#005B82")

    ax.set_yticks([])
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"Neuron")
    
    plt.savefig("plots/input.pgf", transparent=True)
    plt.savefig("plots/input.pdf", transparent=True)
