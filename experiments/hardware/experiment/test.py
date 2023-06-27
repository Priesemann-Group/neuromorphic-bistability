# -*- coding: utf-8 -*-
import os
import json
import argparse
import pylogging
import numpy as np
import helpers as hp
import pystadls_vx_v1 as stadls
import pyhaldls_vx_v1 as haldls
import pyfisch_vx as fisch
import pyhalco_hicann_dls_vx_v1 as halco
import pylola_vx_v1 as lola
import pyhxcomm_vx as hxcomm

import gonzales

from blackbox import HXBlackbox
from blackbox import PPUSignal

CLOCK_CYCLES_PER_S = 1e6 * fisch.fpga_clock_cycles_per_us


def start_generators(builder, freq, seeds=None):
    if seeds is None:
        seeds = np.random.randint(1, 2**22, size=4)
    period = 50
    mask = 0b111111
    source = haldls.BackgroundSpikeSource()
    source.enable_random = True
    source.period = period
    source.mask = mask
    #source.rate = int(freq / 250e6 * 2 ** 8 * period * (mask + 1))
    source.rate = int(np.round(freq / 250e6 * 2 ** 8 * period * (mask + 1), 0))

    if int(np.round(freq / 250e6 * 2 ** 8 * period * (mask + 1), 0))==0: print("ZERO RNGs")

    for i, generator in enumerate(range(4, 8)):
        source.enable = False
        builder.write(halco.BackgroundSpikeSourceOnDLS(generator), source)

        source.enable = True
        source.seed = int(seeds[i])
        source.neuron_label = 32 | (1 << 13) | ((generator % 4) << 11) | ((generator // 4) << 7)
        builder.write(halco.BackgroundSpikeSourceOnDLS(generator), source)

def stop_generators(builder):
    source = haldls.BackgroundSpikeSource()
    source.enable = False
    for i in range(4, 8):
        builder.write(halco.BackgroundSpikeSourceOnDLS(i), source)

def run_experiment(connection, options):
    np.random.seed(options.seed)

    builder = stadls.PlaybackProgramBuilder()

    # initialize weights
    rows = np.arange(halco.SynapseDriverOnDLS.size)
    rows_on_bus = 2 * (rows // 8) + (rows % 2)
    addresses = np.tile(rows_on_bus, (halco.SynapseDriverOnDLS.size, 1)).T

    shape = (halco.SynapseDriverOnDLS.size, halco.NeuronColumnOnDLS.size)
    weights = np.zeros(shape)

    synapse_matrix = lola.SynapseMatrix()
    synapse_matrix.labels.from_numpy(addresses)
    synapse_matrix.weights.from_numpy(weights)
    for synram in halco.iter_all(halco.SynramOnDLS):
        builder.write(synram, synapse_matrix)

    # load PPU program
    elf_file = lola.PPUElfFile(options.program_path)
    elf_symbols = elf_file.read_symbols()
    ppu_signal_coordinate = elf_symbols["command"].coordinate
    ppu_mu_coordinate = elf_symbols["mu"].coordinate
    ppu_sparsity_offset_coordinate = elf_symbols["sparsity_offset"].coordinate
    ppu_update_offset_coordinate = elf_symbols["update_offset"].coordinate
    ppu_random_seed_coordinate = elf_symbols["random_seed"].coordinate
    ppu_sparsity_seed_coordinate = elf_symbols["sparsity_seed"].coordinate
    ppu_ppu_coordinate = elf_symbols["ppu"].coordinate

    program = elf_file.read_program()
    program_on_ppu = halco.PPUMemoryBlockOnPPU(
        halco.PPUMemoryWordOnPPU(0),
        halco.PPUMemoryWordOnPPU(program.size() - 1)
    )

    # ensure PPU is in reset state
    ppu_control_reg = haldls.PPUControlRegister()
    ppu_control_reg.inhibit_reset = False

    for reg in halco.iter_all(halco.PPUControlRegisterOnDLS):
        builder.write(reg, ppu_control_reg)

    # load and ppu program
    for ppu in halco.iter_all(halco.PPUOnDLS):
        program_on_dls = halco.PPUMemoryBlockOnDLS(program_on_ppu, ppu)
        builder.write(program_on_dls, program)

    # update variables
    mu = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(options.mu << 24))
    update_offset = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(options.update_offset << 24))
    sparsity_offset = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(options.sparsity_offset << 24))
    random_seed = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(np.random.randint(1, 0xff, dtype=np.uint8) << 24))
    sparsity_seed = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(np.random.randint(1, 0xff, dtype=np.uint8) << 24))
    for ppu in halco.iter_all(halco.PPUOnDLS):
        ppu_id = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(int(ppu.toEnum()) << 24))
        builder.write(halco.PPUMemoryWordOnDLS(ppu_ppu_coordinate[0], ppu), ppu_id)
        builder.write(halco.PPUMemoryWordOnDLS(ppu_mu_coordinate[0], ppu), mu)
        builder.write(halco.PPUMemoryWordOnDLS(ppu_sparsity_offset_coordinate[0], ppu), sparsity_offset)
        builder.write(halco.PPUMemoryWordOnDLS(ppu_update_offset_coordinate[0], ppu), update_offset)
        builder.write(halco.PPUMemoryWordOnDLS(ppu_random_seed_coordinate[0], ppu), random_seed)
        builder.write(halco.PPUMemoryWordOnDLS(ppu_sparsity_seed_coordinate[0], ppu), sparsity_seed)

    # ensure PPU is in run state
    ppu_control_reg_run = haldls.PPUControlRegister()
    ppu_control_reg_run.inhibit_reset = True

    for reg in halco.iter_all(halco.PPUControlRegisterOnDLS):
        builder.write(reg, ppu_control_reg_run)

    builder.write(halco.TimerOnDLS(), haldls.Timer())
    builder.block_until(halco.TimerOnDLS(),
                       int(options.offset * CLOCK_CYCLES_PER_S))

    for update in range(options.updates):
        # trigger PPU update
        command = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(PPUSignal.RUN.value))
        for ppu in halco.iter_all(halco.PPUOnDLS):
            builder.write(halco.PPUMemoryWordOnDLS(ppu_signal_coordinate[0], ppu), command)

        # wait for PPU to finish
        builder.block_until(halco.TimerOnDLS(),
                           2000*update*fisch.fpga_clock_cycles_per_us + \
                                   int(options.offset * CLOCK_CYCLES_PER_S))

    # ensure PPU is in reset state
    ppu_control_reg_end = haldls.PPUControlRegister()
    ppu_control_reg_end.inhibit_reset = False
    builder.write(halco.PPUControlRegisterOnDLS(), ppu_control_reg_end)

    # read back synram after adaptation
    tickets = list()
    for synram in halco.iter_all(halco.SynramOnDLS):
        tickets.append(builder.read(synram))

    program = builder.done()
    stadls.run(connection, program)

    weights = np.zeros((halco.SynapseRowOnDLS.size,
                        halco.NeuronColumnOnDLS.size), dtype=np.int)
    for i, ticket in enumerate(tickets):
        w = ticket.get().weights.to_numpy()
        weights[i*halco.SynapseRowOnSynram.size:
                (i+1)*halco.SynapseRowOnSynram.size, :] = w

    return weights

def run_static(connection, options):
    np.random.seed(options.seed)

    builder = stadls.PlaybackProgramBuilder()

    # start_generators(builder, options.freq)

    # sync time
    builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
    builder.write(halco.TimerOnDLS(), haldls.Timer())

    builder.block_until(halco.TimerOnDLS(),
                       int(options.offset * CLOCK_CYCLES_PER_S))

    # start event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = True
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    builder.block_until(halco.TimerOnDLS(),
                       int(options.duration * CLOCK_CYCLES_PER_S) + \
                               int(options.offset * CLOCK_CYCLES_PER_S))

    # stop event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = False
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    # stop spike generators
    # stop_generators(builder)

    builder.block_until(
            halco.TimerOnDLS(),
            int((10 * options.duration) * CLOCK_CYCLES_PER_S))

    program = builder.done()
    stadls.run(connection, program)

    return hp.make_spikearray(program.spikes.to_numpy())

def run_pert(connection, options, spikes):
    np.random.seed(options.seed)

    builder = stadls.PlaybackProgramBuilder()

    # start_generators(builder, options.freq)

    # sync time
    builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
    builder.write(halco.TimerOnDLS(), haldls.Timer())

    builder.block_until(halco.TimerOnDLS(),
                       int(options.offset * CLOCK_CYCLES_PER_S))

    # start event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = True
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    # generate burst
    times = spikes[:, 0]
    labels = spikes[:, 1].astype(int)

    busses_to_fire = (labels // 64) % 4
    event_addresses = labels % 64 + 64 * (labels // 128)
    event_addresses |= (1 << 7)
    event_addresses |= (1 << 13)
    event_addresses |= (busses_to_fire << 11)

    gonzales.generate_spiketrain(builder,
                                 times, event_addresses, busses_to_fire)

    builder.block_until(halco.TimerOnDLS(),
                       int((options.duration) * \
                               CLOCK_CYCLES_PER_S))

    # stop event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = False
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    # stop spike generators
    # stop_generators(builder)

    builder.block_until(
            halco.TimerOnDLS(),
            int((10 * options.duration) * CLOCK_CYCLES_PER_S))

    program = builder.done()
    stadls.run(connection, program)

    return hp.make_spikearray(program.spikes.to_numpy())

def run_letter(connection, options, seeds, spikes, freq_rng):
    builder = stadls.PlaybackProgramBuilder()

    builder.write(halco.TimerOnDLS(), haldls.Timer())
    builder.block_until(halco.TimerOnDLS(), 1000 * fisch.fpga_clock_cycles_per_us)

    # sync time
    builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
    builder.write(halco.TimerOnDLS(), haldls.Timer())

    # start event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = True
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    # offset
    builder.block_until(halco.TimerOnDLS(),
                        int(options.offset * CLOCK_CYCLES_PER_S))

    # set seeds
    start_generators(builder, freq_rng, seeds)

    # generate burst
    times = spikes[:, 0]
    labels = spikes[:, 1].astype(int)
    busses_to_fire = (labels // 64) % 4
    event_addresses = labels % 64 + 64 * (labels // 128)
    event_addresses |= (1 << 7)
    event_addresses |= (1 << 13)
    event_addresses |= (busses_to_fire << 11)
    gonzales.generate_spiketrain(builder,
                                 times, event_addresses, busses_to_fire)

    duration = options.duration + options.offset
    builder.block_until(halco.TimerOnDLS(),
                       int(duration * CLOCK_CYCLES_PER_S))

    # stop event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = False
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    program = builder.done()
    stadls.run(connection, program)

    return hp.make_spikearray(program.spikes.to_numpy())


def old_run_letter(connection, options, seeds, pre_letter, post_letter):
    builder = stadls.PlaybackProgramBuilder()

    # builder.write(halco.TimerOnDLS(), haldls.Timer())
    # builder.block_until(halco.TimerOnDLS(), 1000 * fisch.fpga_clock_cycles_per_us)

    # sync time
    builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
    builder.write(halco.TimerOnDLS(), haldls.Timer())

    # start event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = True
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    # offset
    builder.block_until(halco.TimerOnDLS(),
                       int(options.offset * CLOCK_CYCLES_PER_S))

    # pre letter
    start_generators(builder, options.freq, seeds[0, :])
    duration = pre_letter * options.duration + options.offset
    builder.block_until(halco.TimerOnDLS(),
                       int(duration * CLOCK_CYCLES_PER_S))

    # target letter
    start_generators(builder, options.freq, seeds[1, :])
    duration = (pre_letter + 1) * options.duration + options.offset
    builder.block_until(halco.TimerOnDLS(),
                       int(duration * CLOCK_CYCLES_PER_S))

    # post_letter
    start_generators(builder, options.freq, seeds[2, :])
    duration = (pre_letter + post_letter + 1) * options.duration + options.offset
    builder.block_until(halco.TimerOnDLS(),
                       int(duration * CLOCK_CYCLES_PER_S))

    start_generators(builder, options.freq)

    # stop event recording
    event_config = haldls.EventRecordingConfig()
    event_config.enable_event_recording = False
    builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

    program = builder.done()
    stadls.run(connection, program)

    return hp.make_spikearray(program.spikes.to_numpy())




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Stimulation
    parser.add_argument("--freq", type=float, default=20e3)
    parser.add_argument("--duration", type=float, default=0.02)
    parser.add_argument("--offset", type=float, default=10e-6)
    # Homeostasis
    parser.add_argument("--program_path", type=str)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--mu", type=int, default=20)
    # Network
    parser.add_argument("--percentage_inhibitory", type=float, default=0.2)
    parser.add_argument("--sparsity_offset", type=int, default=114)
    parser.add_argument("--update_offset", type=int, default=120)
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

    inhibitory_mask = np.zeros(halco.SynapseRowOnDLS.size, dtype=bool)
    inhibitory_mask[:halco.SynapseRowOnSynram.size] = np.random.random(
            halco.SynapseRowOnSynram.size) < args.percentage_inhibitory

    with hxcomm.ManagedConnection() as conn:
        blackbox = HXBlackbox(conn,
                              inhibitory_mask,
                              use_calibration=args.use_calibration)
        blackbox.initialize()
        blackbox.configure()

        shape = (halco.SynapseRowOnDLS.size, halco.NeuronColumnOnDLS.size)
        weights = np.zeros(shape)

        blackbox.set_weights(weights)

        blackbox.load_and_start_ppu_program(args.program_path,
                                            args.mu,
                                            args.sparsity_offset,
                                            args.update_offset,
                                            34, 85)

        builder = stadls.PlaybackProgramBuilder()
        start_generators(builder, args.freq)
        stadls.run(conn, builder.done())

        input_spikes, network_spikes = [], []

        weights = np.zeros(((args.updates, ) + shape), dtype=np.float)
        for update in range(args.updates):
            log.info(update)
            builder = stadls.PlaybackProgramBuilder()

            command = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(PPUSignal.RUN.value))
            for ppu in halco.iter_all(halco.PPUOnDLS):
                builder.write(halco.PPUMemoryWordOnDLS(blackbox._ppu_signal_coordinate[0], ppu), command)

            builder.write(halco.TimerOnDLS(), haldls.Timer())
            builder.block_until(halco.TimerOnDLS(), 2000*fisch.fpga_clock_cycles_per_us)

            stadls.run(conn, builder.done())
            weights[update, :, :] = blackbox.get_weights()

            #ispikes, nspikes, _ = blackbox.stimulate(args.duration, spike_record=True)
            nspikes, ispikes = run_static(conn, args)
            input_spikes.append(ispikes)
            network_spikes.append(nspikes)

        builder = stadls.PlaybackProgramBuilder()
        stop_generators(builder)
        stadls.run(conn, builder.done())

    if args.prefix:
        args.basedir, args.prefix = os.path.split(args.prefix)
        args.input_spikes_filename = f"{args.prefix}_input_spikes.npy"
        args.spikes_filename = f"{args.prefix}_network_spikes.npy"
        args.weights_filename = f"{args.prefix}_weights.npz"
        # arguments
        with open(os.path.join(args.basedir, f"{args.prefix}_config.json"), "w") as file_handle:
            json.dump(vars(args), file_handle, indent=4)
        # spikes and weights
        np.save(os.path.join(args.basedir, args.input_spikes_filename), input_spikes)
        np.save(os.path.join(args.basedir, args.spikes_filename), network_spikes)
        np.savez(os.path.join(args.basedir, args.weights_filename),
                 weights=weights, inhibitory_mask=inhibitory_mask)
