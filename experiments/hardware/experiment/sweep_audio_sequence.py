import os
import json
import argparse
import pylogging
import numpy as np
import helpers as hp

import pystadls_vx_v1 as stadls
import pyhalco_hicann_dls_vx_v1 as halco
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


class Sequence:
    def __init__(
            self,
            freq_audio,
            args,
            speedup=10, pre=3, post=3, cut=0.8):
        """ get list of pre and post digits """

        self.speedup = speedup
        self.freq_audio = freq_audio
        self.digits_X = args.digits_X

        self.num_samples = 18
        self.num_speakers = 12

        # Figure out the lenght of digit_X
        lengths = []
        for dgt in self.digits_X:
            for spl in range(self.num_samples):
                for spk in range(self.num_speakers):
                    spks = np.load(args.stimuli_path.format(spk, spl, dgt))["arr_0"].T
                    lengths.append(spks[:,0].max())
        lengths = np.array(lengths)
        self.len_digit_X = np.median(lengths) + np.std(lengths)

        # Get the available pre and post digits
        noise_digits = list(set(range(10)) - set(self.digits_X))

        # Draw pre and post digit lists
        pre_digits = np.random.choice(noise_digits, size=pre, replace=True)
        post_digits = np.random.choice(noise_digits, size=post, replace=True)

        # Get spikes for pre and post digits and concat them
        self.prepost = []
        for i, digits in enumerate([pre_digits, post_digits]):
            shift, add = 0, 0
            spikes = []
            # choose random digits spikes
            for digit in digits:
                speaker = np.random.randint(0, self.num_speakers)
                sample = np.random.randint(0, self.num_samples)
                spks = np.load(args.stimuli_path.format(speaker, sample, digit))["arr_0"].T

                # remove tail
                spks = spks[spks[:,0] < cut * spks[:,0].max(),: ]

                # align in time
                add = spks[:, 0].max()
                spks[:, 0] += shift
                shift += add
                spikes.append(spks)
            spikes = np.concatenate(spikes)

            self.prepost.append(spikes)

        self.labels = pre_digits.tolist() + [digit] + post_digits.tolist()
        self.durations = [self.prepost[0][:, 0].max(),
                          self.prepost[0][:, 0].max() + self.len_digit_X,
                          self.prepost[0][:, 0].max() + self.len_digit_X + self.prepost[1][:,0].max()]
        self.durations = [x*1e-3/self.speedup for x in self.durations]

    def _speedup(self, spks):
        """ return speedup and subsample copy of given spiketrain """
        spikes = spks.copy()
        # Increase Rate by speedup
        spikes[:, 0] *= 1e-3 / self.speedup
        # subsample to target rate: uniformly select N=freq*duration many spikes
        mask = np.random.choice(
                spikes.shape[0],
                int(self.freq_audio * 256 * spikes[:,0].max()),
                replace=False)
        return spikes[mask, :]

    def sample_prepost(self):
        """ make pre and post sequences with speedup """
        self.prepost_sample = []
        for spikes in self.prepost.copy():
            self.prepost_sample.append(self._speedup(spikes))

    def make(self, digit_X):
        """ speedup digit_X and concat with prepost sequences """

        assert digit_X in self.digits_X

        # get a random sample of the desired digit
        # TODO: do this systematically with an iterator?
        speaker = np.random.randint(0, self.num_speakers)
        sample = np.random.randint(0, self.num_samples)
        spikes = np.load(args.stimuli_path.format(speaker, sample, digit_X))["arr_0"].T

        # cut digitx at defined length
        spikes = spikes[spikes[:,0]<self.len_digit_X, :]

        # Increase Rate by speedup
        spikes = self._speedup(spikes)

        # order sequence and concatenate
        pre = self.prepost_sample[0].copy()
        post = self.prepost_sample[1].copy()
        spikes[:, 0] += self.durations[0]
        post[:, 0] += self.durations[1]
        sequence = np.concatenate([pre, spikes, post])
        sequence = sequence[sequence[:,0].argsort(), :]
        sequence[:, 0] += options.offset

        return sequence



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
    #parser.add_argument("--classes", type=int, nargs='+', default=[0, 2, 3, 9])
    parser.add_argument("--samples", type=int, default=200)

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

    #options.duration = args.duration
    options.freq_audio = 1e3
    args.digits_X = [1,2,8]

    np.random.seed(12345)

    mysequence = Sequence(options.freq_audio, args,
                speedup=10, pre=5, post=5)
    options.duration = mysequence.durations[-1]
    args.durations = mysequence.durations

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

                if not os.path.isfile(os.path.join(args.basedir, net_spikefile)):

                    # sample sequence
                    mysequence.sample_prepost()

                    freq_rng = options.freq - options.freq_audio
                    assert freq_rng > 0

                    np.random.seed(options.seed)
                    # Generator seeds
                    seeds = np.random.randint(1, 2 ** 22, size=4)


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

                    for i_seq, digit_X in enumerate(args.digits_X):
                        in_spikes.append([])
                        net_spikes.append([])

                        for i_sample in range(args.samples):

                            log.debug("At {}/{}".format(i_seq, i_sample))

                            # Record word
                            # TODO: set seeds?
                            sequence = mysequence.make(digit_X)
                            o, i = run_letter(connection, options, None,
                                              sequence, freq_rng)

                            in_spikes[i_seq].append(i)
                            net_spikes[i_seq].append(o)

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
