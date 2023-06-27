import numpy as np
import pyhalco_hicann_dls_vx_v1 as halco
import pyfisch_vx as fisch
from itertools import product


def shape_iter(shape):
    return product(*[range(n) for n in shape])

def update_weights(weights, rates, decay, drift, sigma):
    n = sigma * np.random.randn(*weights.shape) + drift
    weights = weights + n - (decay * rates[None, :])
    weights[weights < 0] = 0
    weights[weights > 63] = 63
    return weights

def fixed_indegree(input_degree, network_degree):
    input_mask = indegree(int(input_degree * 256), (256, 256))
    network_mask = indegree(int(network_degree * 256), (256, 256))
    return np.concatenate((network_mask, input_mask), axis=0)

def indegree(degree, shape):
    mask = np.zeros(shape, dtype=bool)
    for nrn in range(shape[1]):
        perm = np.random.permutation(shape[0])
        mask[perm[:degree], nrn] = True
    return mask 

def jitter(value, width=10):
    v = np.random.randint(-width, width, size=halco.NeuronConfigOnDLS.size) + value
    return np.clip(v, 0, 1022)

def make_spikearray(spikes):
    times = spikes["chip_time"] / float(fisch.fpga_clock_cycles_per_us) * 1e-6
    labels = spikes["label"]

    # construct masks
    is_input = ((labels >> 13) & 0b1)
    input_mask = (is_input == 1)
    network_mask = (is_input == 0)

    # reconstruct input spike labels
    generator_id = ((labels[input_mask] >> 11) & 0b11)
    neuron_address = (labels[input_mask] & 0b111111)
    
    input_spikes = np.zeros((input_mask.sum(), 2))
    input_spikes[:, 0] = times[input_mask]
    input_spikes[:, 1] = 64 * generator_id + neuron_address
    
    # reconstruct network spike labels
    neuron_labels = labels[network_mask] & 2**14 - 1
    bus = labels[network_mask] >> 14
    neuron_blocks = neuron_labels // 256

    network_spikes = np.zeros((network_mask.sum(), 2))
    network_spikes[:, 0] = times[network_mask]
    network_spikes[:, 1] = neuron_labels - 256 * neuron_blocks + \
            bus * 32 + (128 - 32) * ((neuron_labels % 256).astype(np.int) >= 32)
    return network_spikes, input_spikes
