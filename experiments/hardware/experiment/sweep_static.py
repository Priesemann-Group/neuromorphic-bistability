import os
import json
import argparse
import pylogging
import numpy as np

import pyhalco_hicann_dls_vx as halco
import pyhxcomm_vx as hxcomm

from blackbox import HXBlackbox
from test import run_static



dt = 1e-7
def poisson_spiketrains(duration, freq, num_inputs):

    n = 8
    freq *= n

    num_bins = int(duration // dt)
    spikes = np.random.rand(num_bins) < freq * dt
    spikes = np.repeat(spikes[:, np.newaxis], num_inputs, axis=1)

    mask = np.random.randint(n, size=num_inputs)
    mask[0] = 0
    spikes[:, mask!=0] = False

    spikes = np.array(np.where(spikes > 0), dtype=float)
    spikes[0, :] = spikes[0, :] * dt
    return spikes.T


def run_static2(blackbox, options):
    spikes =  poisson_spiketrains(options.duration, options.freq, num_inputs=256)
    in_spikes, net_spikes, rates, samples = blackbox.stimulate(
            duration=options.duration,
            input_spikes=spikes,
            spike_record=True)
    return net_spikes, in_spikes



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sweep over many stdp experiments")
    # The options file is a json with the keys of the "normal" stdp experiment.
    # Any config.json saved from a stdp.py can serve as a config here.
    parser.add_argument("options", type=str)
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
    
    np.random.seed(890234)
            
    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)
    
    with hxcomm.ManagedConnection() as connection:
        blackbox = HXBlackbox(connection, inh_mask,
                              enable_loopback=True,
                              use_calibration=options.use_calibration)
        blackbox.initialize()
        blackbox.configure()
        
        for index0, value0 in enumerate(args.values0):
            # Initialize return list of result filename for the inner loop
            args.input_spikefiles.append([])
            args.network_spikefiles.append([])
            
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
                args.input_spikefiles[index0].append(in_spikefile)
                args.network_spikefiles[index0].append(net_spikefile)
        
                if not os.path.isfile(os.path.join(args.basedir, net_spikefile)):
                    # Load weights
                    weights = np.load(os.path.join(args.optionsdir, args.weightfiles[index0][index1]))
                    inh_mask = weights["inh_mask"]
                    weights = weights["weights"]
                    blackbox.set_weights(weights)
                    blackbox.set_inhibitory_mask(inh_mask)

                    # Run the expriment
                    net_spikes, in_spikes = run_static(connection, options)

                    np.save(os.path.join(args.basedir, in_spikefile), in_spikes)
                    np.save(os.path.join(args.basedir, net_spikefile), net_spikes)

    args.configfile = f"{args.prefix}_config.json"
    with open(os.path.join(args.basedir, args.configfile), "w") as handle:
        json.dump(vars(args), handle, indent=4)
