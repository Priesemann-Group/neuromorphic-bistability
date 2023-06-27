import os
import json
import argparse
import pylogging
import numpy as np
import helpers as hp

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

fname = "lang-german_speaker-{:02d}_trial-{}_digit-{}.npz"


dt = 1e-7
def poisson_spiketrains(duration, freq, num_inputs):
    num_bins = int(duration // dt)
    spikes = np.random.rand(num_bins, num_inputs) < freq * dt
    spikes = np.array(np.where(spikes > 0), dtype=float)
    spikes[0, :] = spikes[0, :] * dt
    return spikes.T

def mean_freq(args):
    shape = (len(args.classes), args.samples, args.speaker)
    activity = np.zeros(shape)
    for i, j, k in hp.shape_iter(shape):
        spks = np.load(args.stimuli_path.format(k, j, args.classes[i]))["arr_0"]
        activity[i, j, k] = spks.shape[1] / \
                halco.SynapseRowOnSynram.size / spks[0, :].max() / 1e-3
    return activity #.mean(axis=1)

def load_spikes(cls, s, spk, options):
    # TODO: WAIT ENOUGH

    fname = args.stimuli_path.format(spk, s, cls)
    spikes = np.load(fname)["arr_0"].T
    spikes[:, 0] *= 1e-3

    diff = options.freq_audio - spikes.shape[0] / (256 * spikes[:,0].max())
    if diff < 0:
        # uniformly select N=freq*duration many spikes
        mask = np.random.choice(
            spikes.shape[0],
            int(options.freq_audio * 256 * spikes[:,0].max()),
            replace=False)
        spikes = spikes[mask, :]
    else:
        #print(cls, s, spk, "ADDING", diff)
        options.ADDING += 1
        extra_spikes = poisson_spiketrains(
            spikes[:,0].max(),
            diff,
            halco.SynapseRowOnSynram.size)
        spikes = np.concatenate([spikes, extra_spikes])

    spikes[:, 0] = np.sort(spikes[:, 0])
    spikes[:, 0] += options.offset
    return spikes


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
    parser.add_argument("--stimuli_path", type=str)
    parser.add_argument("--classes", type=int, nargs='+', default=[0, 2, 3, 9])
    parser.add_argument("--speaker", type=int, default=12)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--duration", type=float, default=1e-3)

    args = parser.parse_args()
    args.basedir, args.prefix = os.path.split(args.prefix)
    args.stimuli_path = os.path.join(args.stimuli_path, fname)

    # Load the experiment options
    args.options = json.load(args.options)

    # Convert the options to a class
    class Options(object):
        pass
    options = Options()
    options.__dict__.update(args.options)
    options.duration = args.duration

#    options.freq_audio = 2e3
    options.fraction_audio = 0.5

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

                freq_rng = (1-options.fraction_audio)*options.freq
#                freq_rng = options.freq-options.freq_audio
                assert freq_rng > 0
                options.freq_audio = options.fraction_audio*options.freq

                np.random.seed(options.seed)

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

                    options.ADDING = 0

                    for c, cls in enumerate(args.classes):
                        in_spikes.append([])
                        net_spikes.append([])

                        for s in range(args.samples):
                            in_spikes[c].append([])
                            net_spikes[c].append([])

                            for spk in range(args.speaker):
                                log.debug("At {}/{}/{}".format(c, s, spk))

                                spikes = load_spikes(cls, s, spk, options)

                                # Record word
                                o, i = run_letter(connection, options, None,
                                                  spikes, freq_rng)

                                in_spikes[c][s].append(i)
                                net_spikes[c][s].append(o)

                    if True:#index0==0 and index1==0:
                        print("Added spikes in {:.2f}% of cases".format(options.ADDING / len(args.classes) / args.samples / args.speaker *100))

                    np.save(os.path.join(args.basedir, in_spikefile),
                            in_spikes)
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
