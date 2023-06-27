import os
import json
import argparse
import pylogging
import numpy as np

import pystadls_vx as stadls
import pyhalco_hicann_dls_vx as halco
import pyhxcomm_vx as hxcomm

from blackbox import HXBlackbox
from test import run_experiment, run_static, start_generators, stop_generators

type_map = {
        "freq": float,
        "seed": int,
        "lam": int,
        "mu": int,
        "offset": int
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sweep over many stdp experiments")
    # The options file is a json with the keys of the "normal" stdp experiment.
    # Any config.json saved from a stdp.py can serve as a config here.
    parser.add_argument("options", type=argparse.FileType("r"))
    parser.add_argument("prefix", type=str)
    # Sweep
    parser.add_argument("--key0", type=str, required=True)
    parser.add_argument("--key1", type=str, required=True)
    parser.add_argument("--min0", type=float, default=2e3)
    parser.add_argument("--min1", type=float, default=728904)
    parser.add_argument("--max0", type=float, default=90e3)
    parser.add_argument("--max1", type=float, default=729003)
    parser.add_argument("--step0", type=float, default=8)
    parser.add_argument("--step1", type=float, default=1)
    parser.add_argument("--ax0log", action="store_true")
    parser.add_argument("--ax1log", action="store_true")

    args = parser.parse_args()
    args.basedir, args.prefix = os.path.split(args.prefix)

    # Load the experiment options
    args.options = json.load(args.options)

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

    # Create the swept range
    if args.ax0log:
        args.values0 = np.logspace(np.log10(args.min0), np.log10(args.max0),
                                   int(args.step0), dtype=type_map[args.key0])
    else:
        args.values0 = np.arange(args.min0, args.max0 + args.step0,
                                 args.step0, dtype=type_map[args.key0])
    if args.ax1log:
        args.values1 = np.logspace(np.log10(args.min1), np.log10(args.max1),
                                   int(args.step1), dtype=type_map[args.key1])
    else:
        args.values1 = np.arange(args.min1, args.max1 + args.step1,
                                 args.step1, dtype=type_map[args.key1])

    # Sweep over the paramter range
    args.weightfiles = []
    args.input_spikefiles = []
    args.network_spikefiles = []
    
    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)

    with hxcomm.ManagedConnection() as connection:
        blackbox = HXBlackbox(connection, inh_mask,
                              enable_loopback=False,
                              use_calibration=options.use_calibration)
        blackbox.initialize()
        blackbox.configure()

        for index0, value0 in enumerate(args.values0):
            # Initialize return list of result filename for the inner loop
            args.weightfiles.append([])
            args.input_spikefiles.append([])
            args.network_spikefiles.append([])

            # Update the outter loop parameter
            log.INFO("Set {}: {}".format(args.key0, value0))
            setattr(options, args.key0, value0)

            for index1, value1 in enumerate(args.values1):
                # Update the inner loop parameter
                log.INFO("Set {}: {}".format(args.key1, value1))
                setattr(options, args.key1, value1)

                # Run the expriment
                in_spikefile = f"{args.prefix}_input_spikes_{index0:03d}_{index1:03d}.npy"
                net_spikefile = f"{args.prefix}_network_spikes_{index0:03d}_{index1:03d}.npy"
                weightfile = f"{args.prefix}_weights_{index0:03d}_{index1:03d}.npz"
                args.input_spikefiles[index0].append(in_spikefile)
                args.network_spikefiles[index0].append(net_spikefile)
                args.weightfiles[index0].append(weightfile)

                if not os.path.isfile(os.path.join(args.basedir, weightfile)):

                    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)
                    inh_mask[:halco.SynapseRowOnSynram.size] = np.random.random(
                            halco.SynapseRowOnSynram.size) < options.percentage_inhibitory

                    blackbox.set_inhibitory_mask(inh_mask)

                    builder = stadls.PlaybackProgramBuilder()
                    start_generators(builder, options.freq)
                    stadls.run(connection, builder.done())

                    weights = run_experiment(connection, options)
                    net_spikes, in_spikes = run_static(connection, options)

                    builder = stadls.PlaybackProgramBuilder()
                    stop_generators(builder)
                    stadls.run(connection, builder.done())

                    np.save(os.path.join(args.basedir, in_spikefile), in_spikes)
                    np.save(os.path.join(args.basedir, net_spikefile), net_spikes)
                    np.savez(os.path.join(args.basedir, weightfile), weights=weights, inh_mask=inh_mask)

    args.values0 = args.values0.tolist()
    args.values1 = args.values1.tolist()

    args.configfile = f"{args.prefix}_config.json"
    with open(os.path.join(args.basedir, args.configfile), "w") as handle:
        json.dump(vars(args), handle, indent=4)
