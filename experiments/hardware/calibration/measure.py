import sys
import argparse
import numpy as np
import scipy.optimize

import pystadls_vx as stadls
import pyhaldls_vx as haldls
import pyhxcomm_vx as hxcomm

import pylogging
import pyhalco_hicann_dls_vx as halco

from calibrationBlackbox import CalibrationBlackbox


def jitter(values, width=10):
    return values + np.random.randint(-width, width, 512)


def decay(x, a, b, c):
    return a * np.exp(-(x - 10e-6) / b) + c


def measure_v_leak(blackbox, enable_synin=False):
    logger = pylogging.get("v_leak")

    blackbox.set_enable_spiking(False)

    if not enable_synin:
        parameters = {
            halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: np.zeros(512),
            halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.zeros(512)
            }
        blackbox.set_neuron_cells(parameters)

    input_spikes = np.empty((0, 2))
    values = np.zeros(halco.NeuronConfigOnDLS.size)
    for neuron_index in range(halco.NeuronConfigOnDLS.size):
        sys.stdout.write("\rMeasuring neuron {}/512".format(neuron_index))
        blackbox.set_readout(neuron_index)

        _, _, trace = blackbox.stimulate(100e-6, input_spikes, madc_record=True)

        values[neuron_index] = trace[:, 0].mean()
        
    sys.stdout.write("\r")

    return values


def measure_v_syn(blackbox):
    logger = pylogging.get("v_syn_exc")
    
    blackbox.set_enable_spiking(False)

    values = np.zeros(halco.NeuronConfigOnDLS.size)
    for neuron_index in range(halco.NeuronConfigOnDLS.size):
        sys.stdout.write("\rMeasuring neuron {}/512".format(neuron_index))
        blackbox.set_readout(neuron_index)

        input_spikes = np.empty((0, 2))
        _, _, trace = blackbox.stimulate(100e-6, input_spikes, madc_record=True)

        values[neuron_index] = trace[:, 0].mean()
    
    sys.stdout.write("\r")

    return values


def measure_tau_syn(blackbox, kind):
    logger = pylogging.get("tau_syn")

    blackbox.set_enable_spiking(False)
    
    input_spikes = np.empty((2, 2))
    input_spikes[0, :] = np.array([10e-6, 0])
    input_spikes[1, :] = np.array([10e-6, 1])

    values = np.zeros(halco.NeuronConfigOnDLS.size)
    for neuron_index in range(halco.NeuronConfigOnDLS.size):
        sys.stdout.write("\rMeasuring neuron {}/512".format(neuron_index))
        blackbox.set_readout(neuron_index, kind)

        _, _, trace = blackbox.stimulate(100e-6, input_spikes, madc_record=True)

        mask = trace[:, 1] > 11e-6
        guess = [-1, 5e-6, trace[-1, 0]]

        popt, pcov = scipy.optimize.curve_fit(decay,
                                              trace[mask, 1],
                                              trace[mask, 0],
                                              guess)
        values[neuron_index] = popt[1]

    sys.stdout.write("\r")

    return values


def measure_v_thres(blackbox):
    logger = pylogging.get("v_thresh")

    blackbox.set_enable_spiking(True)
        
    input_spikes = np.empty((50, 2))
    input_spikes[:, 0] = np.linspace(10e-6, 15e-6, input_spikes.shape[0])
    input_spikes[:, 1] = 0

    values = np.zeros(halco.NeuronConfigOnDLS.size)
    for neuron_index in range(halco.NeuronConfigOnDLS.size):
        sys.stdout.write("\rMeasuring neuron {}/512".format(neuron_index))
        blackbox.set_readout(neuron_index)

        _, _, trace = blackbox.stimulate(50e-6, input_spikes, madc_record=True)

        values[neuron_index] = trace[:, 0].max()

    sys.stdout.write("\r")

    blackbox.set_enable_spiking(False)

    return values


def measure_tau_m(blackbox):
    logger = pylogging.get("tau_m")
    
    input_spikes = np.empty((0, 2))
    blackbox.set_enable_spiking(True)

    parameters = {
        halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: np.zeros(512),
        halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.zeros(512),
        }
    blackbox.set_neuron_cells(parameters)

    values = np.zeros(halco.NeuronConfigOnDLS.size)
    for neuron_index in range(halco.NeuronConfigOnDLS.size):
        sys.stdout.write("\rMeasuring neuron {}/512".format(neuron_index))
        blackbox.set_readout(neuron_index, "membrane")
        
        neuron_on_block = (neuron_index % 128) + (128 if neuron_index >= 256 else 0)
        block = (neuron_index // 128) % 2
        
        coord = halco.NeuronResetOnDLS(halco.NeuronResetOnNeuronResetBlock(neuron_on_block),
                                       halco.NeuronResetBlockOnDLS(block))

        experiment_builder = stadls.PlaybackProgramBuilder()
        experiment_builder.write(coord, haldls.NeuronReset())

        _, _, trace = blackbox.stimulate(100e-6,
                                         input_spikes,
                                         experiment_builder,
                                         madc_record=True)

        mask = (trace[:, 1] >= 12e-6) & (trace[:, 1] <= 80e-6)
        guess = [-1, 10e-6, trace[-1, 0]]
        
        popt, pcov = scipy.optimize.curve_fit(decay,
                                              trace[mask, 1],
                                              trace[mask, 0],
                                              guess)
        values[neuron_index] = popt[1]

    sys.stdout.write("\r")
    
    blackbox.set_enable_spiking(False)
    
    parameters = {
        halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: np.random.randint(1000, 1020, size=512),
        halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.random.randint(1000, 1020, size=512)
        }
    blackbox.set_neuron_cells(parameters)

    return values


def measure_v_reset(blackbox):
    logger = pylogging.get("v_reset")

    input_spikes = np.empty((0, 2))

    blackbox.set_enable_spiking(True)
    
    values = np.zeros(halco.NeuronConfigOnDLS.size)
    for neuron_index in range(halco.NeuronConfigOnDLS.size):
        sys.stdout.write("\rMeasuring neuron {}/512".format(neuron_index))
        blackbox.set_readout(neuron_index, "membrane")

        neuron_on_block = (neuron_index % 128) + (128 if neuron_index >= 256 else 0)
        block = (neuron_index // 128) % 2

        coord = halco.NeuronResetOnDLS(halco.NeuronResetOnNeuronResetBlock(neuron_on_block),
                                       halco.NeuronResetBlockOnDLS(block))

        experiment_builder = stadls.PlaybackProgramBuilder()
        experiment_builder.wait_until(halco.TimerOnDLS(), 5000)
        experiment_builder.write(coord, haldls.NeuronReset())
        _, _, trace = blackbox.stimulate(60e-6, input_spikes,
                                         experiment_builder, madc_record=True)

        mask = (trace[:, 1] >= 42e-6) & (trace[:, 1] <= 50e-6)
        values[neuron_index] = np.median(trace[mask, 0])
        
    sys.stdout.write("\r")

    blackbox.set_enable_spiking(False)

    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", type=int, default=68)
    parser.add_argument("--prefix", type=str, default="data")
    args = parser.parse_args()

    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')
    logger = pylogging.get("measure")

    with hxcomm.ManagedConnection() as connection:
        blackbox = CalibrationBlackbox(connection)
        blackbox.initialize()
        blackbox.configure()

        weights_top = np.zeros((128, 256), dtype=np.int8)
        weights_top[0, :] = 63
        weights_top[1, :] = -63
        weights_top[2, :] = 63
        weights_top[3, :] = -63
        weights_bot = np.zeros((128, 256), dtype=np.int8)
        weights_bot[0, :] = 63
        weights_bot[1, :] = -63
        weights_bot[2, :] = 63
        weights_bot[3, :] = -63
        blackbox.set_weights(weights_top, weights_bot)

        params = {
                halco.CapMemRowOnCapMemBlock.v_leak: "{}/v_leak_{}.npy".format(args.prefix, args.chip),
                halco.CapMemRowOnCapMemBlock.i_bias_leak: "{}/tau_m_{}.npy".format(args.prefix, args.chip),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_res: "{}/tau_syn_exc_{}.npy".format(args.prefix, args.chip),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_res: "{}/tau_syn_inh_{}.npy".format(args.prefix, args.chip),
                halco.CapMemRowOnCapMemBlock.v_threshold: "{}/v_thres_{}.npy".format(args.prefix, args.chip),
                halco.CapMemRowOnCapMemBlock.v_syn_exc: "{}/v_syn_exc_{}.npy".format(args.prefix, args.chip),
                halco.CapMemRowOnCapMemBlock.v_syn_inh: "{}/v_syn_inh_{}.npy".format(args.prefix, args.chip),
                halco.CapMemRowOnCapMemBlock.v_reset: "{}/v_reset_{}.npy".format(args.prefix, args.chip)
                }

        for key, value in params.items():
            try:
                blackbox.set_neuron_cells({key: np.load(value)})
            except IOError:
                logger.warn("No values found for {}".format(key))

        values = measure_tau_syn(blackbox, "exc_synin")
        jalsdf

        np.save("{}/tau_syn_exc_calibrated_{}.npy".format(args.prefix, args.chip),
                measure_tau_syn(blackbox, "exc_synin"))
        np.save("{}/tau_syn_inh_calibrated_{}.npy".format(args.prefix, args.chip),
                measure_tau_syn(blackbox, "inh_synin"))
        np.save("{}/tau_m_calibrated_{}.npy".format(args.prefix, args.chip),
                measure_tau_m(blackbox))
        np.save("{}/v_leak_calibrated_{}.npy".format(args.prefix, args.chip),
                measure_v_leak(blackbox, enable_synin=True))
        np.save("{}/v_thres_calibrated_{}.npy".format(args.prefix, args.chip),
                measure_v_thres(blackbox))
        np.save("{}/v_reset_calibrated_{}.npy".format(args.prefix, args.chip),
                measure_v_reset(blackbox))
        
        for key, value in params.items():
            try:
                blackbox.set_neuron_cells({key: jitter(np.load(value).mean().astype(np.int))})
            except IOError:
                logger.warn("No values found for {}".format(key))
        
        np.save("{}/tau_syn_exc_uncalibrated_{}.npy".format(args.prefix, args.chip),
                measure_tau_syn(blackbox, "exc_synin"))
        np.save("{}/tau_syn_inh_uncalibrated_{}.npy".format(args.prefix, args.chip),
                measure_tau_syn(blackbox, "inh_synin"))
        np.save("{}/tau_m_uncalibrated_{}.npy".format(args.prefix, args.chip),
                measure_tau_m(blackbox))
        np.save("{}/v_leak_uncalibrated_{}.npy".format(args.prefix, args.chip),
                measure_v_leak(blackbox, enable_synin=True))
        np.save("{}/v_thres_uncalibrated_{}.npy".format(args.prefix, args.chip),
                measure_v_thres(blackbox))
        np.save("{}/v_reset_uncalibrated_{}.npy".format(args.prefix, args.chip),
                measure_v_reset(blackbox))
