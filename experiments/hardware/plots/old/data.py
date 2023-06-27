import numpy as np
import matplotlib.cm as cm


class Standard:
    speedup = 1000
    Nneurons = 256
    Ninputs = 256
    
    freqs = np.array([4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3, 50e3, 55e3],
                     dtype=int)
    seeds = np.arange(10)

    Nfreqs = freqs.size
    Nseeds = seeds.size

    shape = (Nfreqs, Nseeds)

    # network_spike_path = "../experiments/data/sweep_{}_{}_network_spikes.npy"
    # input_spike_path = "../experiments/data/sweep_{}_{}_input_spikes.npy"
    # weights_path = "../experiments/data/sweep_{}_{}_weights.npz"
    # 
    # network_spike_path_uncalib = "../experiments/data/sweep_uncalib_{}_{}_network_spikes.npy"
    # input_spike_path_uncalib = "../experiments/data/sweep_uncalib_{}_{}_input_spikes.npy"
    # weights_path_uncalib = "../experiments/data/sweep_uncalib_{}_{}_weights.npz"
    
    network_spike_path = "../experiments/data/debug_{}_{}_network_spikes.npy"
    input_spike_path = "../experiments/data/debug_{}_{}_input_spikes.npy"
    weights_path = "../experiments/data/debug_{}_{}_weights.npz"
    
    network_spike_path_uncalib = "../experiments/data/debug_uncalib_{}_{}_network_spikes.npy"
    input_spike_path_uncalib = "../experiments/data/debug_uncalib_{}_{}_input_spikes.npy"
    weights_path_uncalib = "../experiments/data/debug_uncalib_{}_{}_weights.npz"
    
    paths = [network_spike_path, network_spike_path_uncalib]
    Npaths = len(paths)

    tmin = 0
    tmax = 1e-1

    binwidth = 1e-6
    bins = np.arange(tmin, tmax, binwidth)
    steps = np.arange(1, 500)
    Nsteps = steps.size
    
    colors = cm.viridis(np.linspace(0, 1, Nfreqs))

class Sigma:
    speedup = 1000
    Nneurons = 256
    Ninputs = 256
    
    sigmas = np.array([0.2, 0.4, 0.6, 0.8])
    seeds = np.arange(10)

    Nsigmas = sigmas.size
    Nseeds = seeds.size

    shape = (Nsigmas, Nseeds)

    network_spike_path = "../experiments/data/sigma_{}_{}_network_spikes.npy"
    input_spike_path = "../experiments/data/sigma_{}_{}_input_spikes.npy"
    weights_path = "../experiments/data/sigma_{}_{}_weights.npz"
    
    network_spike_path_uncalib = "../experiments/data/sigma_uncalib_{}_{}_network_spikes.npy"
    input_spike_path_uncalib = "../experiments/data/sigma_uncalib_{}_{}_input_spikes.npy"
    weights_path_uncalib = "../experiments/data/sigma_uncalib_{}_{}_weights.npz"
    
    paths = [network_spike_path, network_spike_path_uncalib]
    Npaths = len(paths)

    tmin = 0
    tmax = 1e-1

    binwidth = 1e-6
    bins = np.arange(tmin, tmax, binwidth)
    steps = np.arange(1, 500)
    Nsteps = steps.size
    
    colors = cm.viridis(np.linspace(0, 1, Nsigmas))
