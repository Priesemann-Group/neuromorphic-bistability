# -*- coding: utf-8 -*-
import json
import argparse
import pylogging
import numpy as np
import helpers as hp
import pyhaldls_vx as haldls
import pyhalco_hicann_dls_vx as halco

from blackbox import HXBlackbox

import matplotlib.pyplot as plt

def pre_hook(builder):
    for counter_reset in halco.iter_all(halco.SpikeCounterResetOnDLS):
        builder.write(counter_reset, haldls.SpikeCounterReset())

    period = 20
    mask = 0b111111
    rate = int(args.freq / 250e6 * 2 ** 8 * period * (mask + 1))
        
    source = haldls.BackgroundSpikeSource()
    source.enable = True
    source.enable_random = True # enable poisson and random addresses
    source.rate = rate # probability (8 bit) to fire in a bin
    source.period = period # binning period
    source.mask = mask # bits of address to randomize

    for i in range(4, 8):
        source.seed = np.random.randint(1, 22 ** 2) # seed must not be 0
        source.neuron_label = 32 | (1 << 13) | ((i % 4) << 11) | ((i // 4) << 7) # base address
        builder.write(halco.BackgroundSpikeSourceOnDLS(i), source)

def post_hook(builder):
    source = haldls.BackgroundSpikeSource()
    source.enable = False 
    for i in range(4, 8):
        builder.write(halco.BackgroundSpikeSourceOnDLS(i), source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Network architecture
    parser.add_argument("--percentage_inhibitory", type=float, default=0.2)
    parser.add_argument("--input_degree", type=float, default=0.05)
    parser.add_argument("--network_degree", type=float, default=0.05)
    # Stimulation
    parser.add_argument("--duration", type=float, default=1e-3)
    parser.add_argument("--static_duration", type=float, default=100e-3)
    parser.add_argument("--freq", type=float, default=20e3)
    # Homeostasis
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--decay", type=float, default=0.00003)
    parser.add_argument("--drift", type=float, default=0.6)
    parser.add_argument("--sigma", type=float, default=0.6)
    # calibration
    parser.add_argument("--use_calibration", action="store_true")
    # Seed
    parser.add_argument("--seed", type=int, default=12345)
    # Save
    parser.add_argument("--prefix", type=str, default="debug")
    args = parser.parse_args()

    # Setup logging
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')
    log = pylogging.get("wavy")

    np.random.seed(args.seed)

    inhibitory_mask = np.random.random(512) < args.percentage_inhibitory
    weight_mask = hp.fixed_indegree(args.input_degree, args.network_degree)
    
    blackbox = HXBlackbox(inhibitory_mask, use_calibration=args.use_calibration)
    blackbox.initialize()
    blackbox.configure()
    
    weights = np.zeros((args.updates, 512, 256), dtype=np.float)
    
    for i in range(1, args.updates):
        blackbox.set_weights(weights[i-1, :, :])
        _, _, rates, _ = blackbox.stimulate(duration=args.duration,
        				    pre_hook=pre_hook,
        				    post_hook=post_hook)
        log.info("Rate at update {}/{} is {:.3f}".format(i, args.updates,
                                                         rates.mean() / 1000.))

        weights[i, :, :] = hp.update_weights(weights[i-1, :, :],
                                             rates,
                                             args.decay,
                                             args.drift,
                                             args.sigma) * weight_mask
    blackbox.set_weights(weights[-1, :, :])
    blackbox.set_readout(0)
    
    log.info("Static experiment")
    input_spikes, network_spikes, _, madc_samples = blackbox.stimulate(duration=args.static_duration,
                                                            spike_record=True,
                                                            madc_record=True,
                                                            pre_hook=pre_hook,
                                                            post_hook=post_hook)
    
    np.save("trace.npy", madc_samples)
    if args.prefix:
        with open("{}_config.json".format(args.prefix), "w") as file_handle:
            json.dump(vars(args), file_handle, indent=4)
        np.save("{}_input_spikes.npy".format(args.prefix), input_spikes)
        np.save("{}_network_spikes.npy".format(args.prefix), network_spikes)
        np.savez("{}_weights.npz".format(args.prefix),
                weights=weights[-1, :, :], inhibitory_mask=inhibitory_mask)
