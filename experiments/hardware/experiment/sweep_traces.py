import os
import json
import argparse
import pylogging
import numpy as np

import pyhalco_hicann_dls_vx as halco
import pyhxcomm_vx as hxcomm
import pyhaldls_vx as haldls

from blackbox import HXBlackbox


def pre_hook(builder):
    period = 20
    mask = 0b111111
    rate = int(options.freq / 250e6 * 2 ** 8 * period * (mask + 1))
        
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
    parser = argparse.ArgumentParser("Sweep over many stdp experiments")
    # The options file is a json with the keys of the "normal" stdp experiment.
    # Any config.json saved from a stdp.py can serve as a config here.
    parser.add_argument("options", type=str)
    parser.add_argument("--record_neuron", type=int, default=0)
    parser.add_argument("--record_target", type=str, default="membrane")
    parser.add_argument("prefix", type=str)

    args = parser.parse_args()
    args.basedir, args.prefix = os.path.split(args.prefix)

    # Load the experiment options
    args.options = os.path.abspath(args.options)
    args.optionsdir = os.path.dirname(args.options)
    with open(args.options) as h:
        options = json.load(h)
    args.key0 = options["key0"]
    args.key1 = options["key1"]
    args.values0 = options["values0"]
    args.values1 = options["values1"]
    args.options = options["options"]
    args.weightfiles = options["weightfiles"]

    # Convert the options to a class
    class Options(object):
        pass
    options = Options()
    options.__dict__.update(args.options)

    # Setup logging
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')
    log = pylogging.get("main")

    # Sweep over the paramter range
    args.input_spikefiles = []
    args.network_spikefiles = []
    args.samplefiles = []
    
    np.random.seed(890234)
            
    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)
    
    with hxcomm.ManagedConnection() as connection:
        blackbox = HXBlackbox(connection, inh_mask,
                              enable_loopback=True,
                              use_calibration=options.use_calibration)
        blackbox.initialize()
        blackbox.configure()
        blackbox.set_readout(args.record_neuron, args.record_target)
        
        for index0, value0 in enumerate(args.values0):
            # Initialize return list of result filename for the inner loop
            args.input_spikefiles.append([])
            args.network_spikefiles.append([])
            args.samplefiles.append([])
            
            # Update the outter loop parameter
            log.INFO("Set {}: {}".format(args.key0, value0))
            setattr(options, args.key0, value0)

            for index1, value1 in enumerate(args.values1):
                # Update the inner loop parameter
                log.INFO("Set {}: {}".format(args.key1, value1))
                setattr(options, args.key1, value1)

                # Initilaize file names
                in_spikefile = f"{args.prefix}_input_spikes_{index0:03d}_{index1:03d}.npy"
                net_spikefile = f"{args.prefix}_network_spikes_{index0:03d}_{index1:03d}.npy"
                samplefile = f"{args.prefix}_{args.record_target}_{index0:03d}_{index1:03d}.npy"
                args.input_spikefiles[index0].append(in_spikefile)
                args.network_spikefiles[index0].append(net_spikefile)
                args.samplefiles[index0].append(samplefile)
        
                if not os.path.isfile(os.path.join(args.basedir, samplefile)):
                    # Load weights
                    weights = np.load(os.path.join(args.optionsdir,
                                      args.weightfiles[index0][index1]))
                    inh_mask = weights["inh_mask"]
                    weights = weights["weights"]
                    blackbox.set_weights(weights)
                    blackbox.set_inhibitory_mask(inh_mask)

                    # Run the expriment
                    in_spikes, net_spikes, samples = blackbox.stimulate(options.duration,
                                                                        spike_record=True,
                                                                        madc_record=True,
                                                                        pre_hook=pre_hook,
                                                                        post_hook=post_hook)
                                                    
                    np.save(os.path.join(args.basedir, in_spikefile), in_spikes)
                    np.save(os.path.join(args.basedir, net_spikefile), net_spikes)
                    np.save(os.path.join(args.basedir, samplefile), samples)

    args.configfile = f"{args.prefix}_config.json"
    with open(os.path.join(args.basedir, args.configfile), "w") as handle:
        json.dump(vars(args), handle, indent=4)
