import os
import json
import argparse
import pylogging
import numpy as np

import pystadls_vx as stadls
import pyhalco_hicann_dls_vx as halco
import pyhxcomm_vx as hxcomm

from blackbox import HXBlackbox
from test import run_experiment, run_pert, start_generators, stop_generators

INPUT = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sweep over many stdp experiments")
    # The options file is a json with the keys of the "normal" stdp experiment.
    # Any config.json saved from a stdp.py can serve as a config here.
    parser.add_argument("options", type=str)
    parser.add_argument("prefix", type=str)
    # parser.add_argument("--nspikes", type=int, default=15)
    parser.add_argument("--duration", type=float, default=2.0e-3)
    parser.add_argument("--perttime", type=float, default=1.3e-3)

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
    options.duration = args.duration
    options.perttime = args.perttime

    options.offset = 1e-3

    # Setup logging
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        #date_format='RELATIVE'
        )
    log = pylogging.get("main")

    # Sweep over the paramter range
    if INPUT: args.input_spikefiles = []
    args.network_spikefiles = []

    np.random.seed(890234)

    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)

    with hxcomm.ManagedConnection() as connection:
        blackbox = HXBlackbox(connection, inh_mask,
                              enable_loopback=INPUT,
                              use_calibration=options.use_calibration)
        blackbox.initialize()
        blackbox.configure()

        for index0, value0 in enumerate(args.values0):
            # Initialize return list of result filename for the inner loop
            if INPUT: args.input_spikefiles.append([])
            args.network_spikefiles.append([])

            # Update the outter loop parameter
            log.INFO("Set {}: {}".format(args.key0, value0))
            setattr(options, args.key0, value0)

            for index1, value1 in enumerate(args.values1):
                # Update the inner loop parameter
                log.INFO("Set {}: {}".format(args.key1, value1))
                setattr(options, args.key1, value1)

                # Initilaize file names
                if INPUT: in_spikefile = f"{args.prefix}_input_spikes_{index0:03d}_{index1:03d}.npy"
                net_spikefile = f"{args.prefix}_network_spikes_{index0:03d}_{index1:03d}.npy"
                if INPUT: args.input_spikefiles[index0].append(in_spikefile)
                args.network_spikefiles[index0].append(net_spikefile)

                if not os.path.isfile(os.path.join(args.basedir, net_spikefile)):

                    # # Load weights
                    # weights = np.load(os.path.join(args.optionsdir, args.weightfiles[index0][index1]))
                    # inh_mask = weights["inh_mask"]
                    # weights = weights["weights"]
                    # blackbox.set_weights(weights)
                    # blackbox.set_inhibitory_mask(inh_mask)

                    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)
                    inh_mask[:halco.SynapseRowOnSynram.size] = np.random.random(
                            halco.SynapseRowOnSynram.size) < options.percentage_inhibitory
                    blackbox.set_inhibitory_mask(inh_mask)

                    builder = stadls.PlaybackProgramBuilder()
                    start_generators(builder, options.freq)
                    stadls.run(connection, builder.done())

                    weights = run_experiment(connection, options)

                    # Run the experiment
                    if INPUT: in_spikes = list()
                    net_spikes = list()

                    for index2 in range(halco.NeuronColumnOnDLS.size):

                        # new seed for every neuron
                        setattr(options, args.key1, args.values1[-1]+1 + index1*256+index2)

                        # log.INFO(index2)
                        spikes = np.zeros((1, 2))
                        spikes[:, 0] = args.perttime #+ options.offset
                        spikes[:, 1] = index2

                        # Run the expriment
                        o, i = run_pert(connection, options, spikes)
                        #o = o[o[:,0]>1e-3,:]

                        if INPUT: in_spikes.append(i)
                        net_spikes.append(o)

                    builder = stadls.PlaybackProgramBuilder()
                    stop_generators(builder)
                    stadls.run(connection, builder.done())

                    if INPUT: np.save(os.path.join(args.basedir, in_spikefile), in_spikes)
                    np.save(os.path.join(args.basedir, net_spikefile), net_spikes)

    args.configfile = f"{args.prefix}_config.json"
    with open(os.path.join(args.basedir, args.configfile), "w") as handle:
        json.dump(vars(args), handle, indent=4)
