import os
import json
import argparse
import pylogging
import numpy as np

import pystadls_vx as stadls
import pyhalco_hicann_dls_vx as halco
import pyhxcomm_vx as hxcomm

from blackbox import HXBlackbox
from test import run_experiment, run_letter, start_generators, stop_generators


type_map = {
        "freq": float,
        "seed": int,
        "lam": int,
        "mu": int,
        "offset": int
        }

dt = 1e-7

def poisson_spiketrains(duration, freq, num_inputs):
    num_bins = int(duration // dt)
    spikes = np.random.rand(num_bins, num_inputs) < freq * dt
    spikes = np.array(np.where(spikes > 0), dtype=float)
    spikes[0, :] = spikes[0, :] * dt
    return spikes.T


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
    # Letter options
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--duration", type=float, default=40e-6)
    parser.add_argument("--pre_letter", type=int, default=20)
    parser.add_argument("--post_letter", type=int, default=15)
    parser.add_argument("--n_overlay", type=int, default=5)
    parser.add_argument("--sigma_t", type=float, default=0.0)
    parser.add_argument("--sigma_n", type=float, default=0.0)

    args = parser.parse_args()
    args.basedir, args.prefix = os.path.split(args.prefix)

    # Load the experiment options
    args.options = json.load(args.options)

    # Convert the options to a class
    class Options(object):
        pass
    options = Options()
    options.__dict__.update(args.options)
    options.duration = (args.pre_letter + args.post_letter + 1) * args.duration

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
    args.weightfiles = list()
    args.input_spikefiles = list()
    args.network_spikefiles = list()

    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)

    num_inputs = halco.SynapseRowOnSynram.size

    freq_overlay = args.n_overlay / \
            halco.SynapseRowOnSynram.size / args.duration

    with hxcomm.ManagedConnection() as connection:
        blackbox = HXBlackbox(connection, inh_mask,
                              enable_loopback=True,
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

                # Initilaize file names
                in_spikefile = f"{args.prefix}_input_spikes_{index0:03d}_{index1:03d}.npy"
                net_spikefile = f"{args.prefix}_network_spikes_{index0:03d}_{index1:03d}.npy"
                weightfile = f"{args.prefix}_weights_{index0:03d}_{index1:03d}.npz"
                args.input_spikefiles[index0].append(in_spikefile)
                args.network_spikefiles[index0].append(net_spikefile)
                args.weightfiles[index0].append(weightfile)

                np.random.seed(options.seed)
                # Generator seeds
                seeds = np.random.randint(1, 2 ** 22, size=4)

                if not os.path.isfile(os.path.join(args.basedir, net_spikefile)):

                    inh_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)
                    inh_mask[:halco.SynapseRowOnSynram.size] = np.random.random(
                            halco.SynapseRowOnSynram.size) < options.percentage_inhibitory

                    blackbox.set_inhibitory_mask(inh_mask)

                    builder = stadls.PlaybackProgramBuilder()
                    start_generators(builder, options.freq)
                    stadls.run(connection, builder.done())

                    weights = run_experiment(connection, options)

                    # Run the expriment
                    in_spikes = list()
                    net_spikes = list()

                    freq_rng = options.freq - freq_overlay

                    assert freq_rng >= 0
                    if index1 == 0:
                        log.INFO("RNGs:", freq_rng)
                        log.INFO("Overlay:", freq_overlay)

                    pre_spikes = poisson_spiketrains(
                            args.pre_letter * args.duration,
                            freq_overlay,
                            halco.SynapseRowOnSynram.size)
                    extra_spikes = poisson_spiketrains(
                            args.duration,
                            freq_overlay,
                            halco.SynapseRowOnSynram.size)
                    post_spikes = poisson_spiketrains(
                            args.post_letter * args.duration,
                            freq_overlay,
                            halco.SynapseRowOnSynram.size)
                    pre_spikes[:, 0] += options.offset
                    extra_spikes[:, 0] += options.offset
                    extra_spikes[:, 0] += args.pre_letter * args.duration
                    post_spikes[:, 0] += options.offset
                    post_spikes[:, 0] += (args.pre_letter + 1) * args.duration


                    for c in range(args.classes):
                        log.debug("At class {}/{}".format(c + 1, args.classes))
                        in_spikes.append([])
                        net_spikes.append([])

                        if c > 0:
                        # A' extra spikes
                        # add a gaussian neuron and time shift with mu depending on complexity
                            extra_spikes[:, 0] += np.random.normal(loc=0, scale=args.sigma_t, size=extra_spikes.shape[0])
                            extra_spikes[:, 0] = np.clip(
                                    extra_spikes[:, 0],
                                    options.offset + (args.pre_letter + 0) * args.duration,
                                    options.offset + (args.pre_letter + 1) * args.duration)
                            extra_spikes[:, 0] = np.sort(extra_spikes[:, 0])
                            extra_spikes[:,1] += np.random.normal(loc=0, scale=args.sigma_n, size=extra_spikes.shape[0])
                            extra_spikes[:,1] %= 256

                        spikes = np.concatenate([pre_spikes,
                                                 extra_spikes,
                                                 post_spikes])

                        for s in range(args.samples):
                            log.debug("At sample {}/{}".format(s + 1,
                                                               args.samples))

                            # Record letter
                            o, i = run_letter(connection, options, seeds,
                                              spikes, freq_rng)

                            in_spikes[c].append(i)
                            net_spikes[c].append(o)

                    np.save(os.path.join(args.basedir, in_spikefile), in_spikes)
                    np.save(os.path.join(args.basedir, net_spikefile),
                            net_spikes)
                    np.savez(os.path.join(args.basedir, weightfile),
                            weights=weights, inh_mask=inh_mask)

                    builder = stadls.PlaybackProgramBuilder()
                    stop_generators(builder)
                    stadls.run(connection, builder.done())

    args.values0 = args.values0.tolist()
    args.values1 = args.values1.tolist()

    args.configfile = f"{args.prefix}_config.json"
    with open(os.path.join(args.basedir, args.configfile), "w") as handle:
        json.dump(vars(args), handle, indent=4)
