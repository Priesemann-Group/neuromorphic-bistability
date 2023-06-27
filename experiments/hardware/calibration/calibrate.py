import sys
import argparse
import numpy as np

import pylogging
import pyhalco_hicann_dls_vx as halco
import pyhxcomm_vx as hxcomm

import measure
from measure import jitter
from calibrationBlackbox import CalibrationBlackbox


def calibrate_tau_m(blackbox):
    logger = pylogging.get("calibrate_tau_m")

    n_iterations = 20
    alpha = 4e6
    target = 10e-6

    tau_m = np.random.randint(100, 120, size=halco.NeuronConfigOnDLS.size)
    for i in range(n_iterations):
        # measure 
        parameters = {halco.CapMemRowOnCapMemBlock.i_bias_leak: tau_m}
        blackbox.set_neuron_cells(parameters)

        values = measure.measure_tau_m(blackbox)
        logger.info("{}: tau_m={}+/-{}".format(i, values.mean(), values.std()))

        # update parameter
        tau_m -= np.clip((alpha * (target - values)).astype(np.int), -10, 10)
        tau_m = np.clip(tau_m, 0, 1022)

    return tau_m


def calibrate_tau_syn(blackbox):
    logger = pylogging.get("calibrate_tau_syn_exc")

    n_iterations = 20
    alpha = 5e6
    target = 6e-6

    tau_syn_exc = np.random.randint(80, 100, size=halco.NeuronConfigOnDLS.size)
    tau_syn_inh = np.random.randint(50, 70, size=halco.NeuronConfigOnDLS.size)
    for i in range(n_iterations):
        # measure 
        parameters = {
            halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_res: tau_syn_exc,
            halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_res: tau_syn_inh
            }
        blackbox.set_neuron_cells(parameters)

        values_exc = measure.measure_tau_syn(blackbox, "exc_synin")
        logger.info("{}: tau_syn_exc={}+/-{}".format(i,
                                                     values_exc.mean(),
                                                     values_exc.std()))
        values_inh = measure.measure_tau_syn(blackbox, "inh_synin")
        logger.info("{}: tau_syn_inh={}+/-{}".format(i,
                                                     values_inh.mean(),
                                                     values_inh.std()))
        # update parameter
        tau_syn_exc -= np.clip((alpha * (target - values_exc)).astype(np.int), -10, 10)
        tau_syn_inh -= np.clip((alpha * (target - values_inh)).astype(np.int), -10, 10)

        tau_syn_exc = np.clip(tau_syn_exc, 0, 1022)
        tau_syn_inh = np.clip(tau_syn_inh, 0, 1022)

    return tau_syn_exc, tau_syn_inh


def calibrate_v_leak(blackbox, enable_synin=False):
    logger = pylogging.get("calibrate_v_leak")

    n_iterations = 20
    alpha = 0.5
    target = 400

    v_leak = np.random.randint(780, 800, size=halco.NeuronConfigOnDLS.size)
    for i in range(n_iterations):
        # measure 
        parameters = {halco.CapMemRowOnCapMemBlock.v_leak: v_leak}
        blackbox.set_neuron_cells(parameters)

        values = measure.measure_v_leak(blackbox, enable_synin)
        logger.info("{}: v_leak={}+/-{}".format(i,
                                                values.mean(),
                                                values.std()))

        # update parameter
        v_leak += np.clip((alpha * (target - values)).astype(np.int), -10, 10)
        v_leak = np.clip(v_leak, 0, 1022)

    return v_leak


def calibrate_v_syn_exc(blackbox):
    logger = pylogging.get("calibrate_v_syn_exc")

    n_iterations = 20
    alpha = 0.04

    logger.info("Measuring target value")
    target = measure.measure_v_leak(blackbox, enable_synin=False)

    v_syn_exc = np.random.randint(750, 770, size=halco.NeuronConfigOnDLS.size)
    for i in range(n_iterations):
        # measure
        parameters = {
            halco.CapMemRowOnCapMemBlock.v_syn_exc: v_syn_exc,
            halco.CapMemRowOnCapMemBlock.v_syn_inh: jitter(np.zeros(512) + 650),
            halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: jitter(np.zeros(512) + 1000),
            halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.zeros(512),
            }
        blackbox.set_neuron_cells(parameters)

        values = measure.measure_v_syn(blackbox)
        distance = target - values
        logger.info("{}: distance={}+/-{}".format(i,
                                                  np.abs(distance).mean(),
                                                  np.abs(distance).std()))

        # update parameter
        v_syn_exc += np.clip((alpha * distance).astype(np.int), -10, 10)
        v_syn_exc = np.clip(v_syn_exc, 0, 1022)

    return v_syn_exc


def calibrate_v_syn_inh(blackbox):
    logger = pylogging.get("calibrate_v_syn_inh")

    n_iterations = 20
    alpha = 0.04

    logger.info("Measuring target value")
    target = measure.measure_v_leak(blackbox, enable_synin=False)

    v_syn_inh = np.random.randint(750, 770, size=halco.NeuronConfigOnDLS.size)
    for i in range(n_iterations):
        # measure 
        parameters = {
            halco.CapMemRowOnCapMemBlock.v_syn_inh: v_syn_inh,
            halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: jitter(np.zeros(512) + 1000),
            }
        blackbox.set_neuron_cells(parameters)

        values = measure.measure_v_syn(blackbox)
        distance = target - values
        logger.info("{}: distance={}+/-{}".format(i,
                                                  np.abs(distance).mean(),
                                                  np.abs(distance).std()))

        # update parameter
        v_syn_inh -= np.clip((alpha * distance).astype(np.int), -10, 10)
        v_syn_inh = np.clip(v_syn_inh, 0, 1022)

    return v_syn_inh


def calibrate_v_thres(blackbox):
    logger = pylogging.get("calibrate_v_thres")

    n_iterations = 20
    alpha = 0.2
    target = 500

    v_thres = np.random.randint(370, 390, size=halco.NeuronConfigOnDLS.size)
    for i in range(n_iterations):
        # measure 
        parameters = {halco.CapMemRowOnCapMemBlock.v_threshold: v_thres}
        blackbox.set_neuron_cells(parameters)

        values = measure.measure_v_thres(blackbox)
        logger.info("{}: v_thres={}+/-{}".format(i,
                                                 values.mean(),
                                                 values.std()))

        # update parameter
        v_thres += np.clip((alpha * (target - values)).astype(np.int), -10, 10)
        v_thres = np.clip(v_thres, 0, 1022)

    return v_thres


def calibrate_v_reset(blackbox):
    logger = pylogging.get("calibrate_v_reset")

    n_iterations = 20
    alpha = 0.4
    target = 350

    v_reset = np.random.randint(680, 700, size=halco.NeuronConfigOnDLS.size)
    for i in range(n_iterations):
        # measure 
        parameters = {halco.CapMemRowOnCapMemBlock.v_reset: v_reset}
        blackbox.set_neuron_cells(parameters)

        values = measure.measure_v_reset(blackbox)
        logger.info("{}: v_reset={}+/-{}".format(i,
                                                 values.mean(),
                                                 values.std()))

        # update parameter
        v_reset += np.clip((alpha * (target - values)).astype(np.int), -10, 10)
        v_reset = np.clip(v_reset, 0, 1022)

    return v_reset

def calibrate_post(blackbox, args):
    logger= pylogging.get("calibrate_post")

    n_iterations = 3
    v_reset_alpha = 0.4
    v_reset_target = 350

    v_thres_alpha = 0.4
    v_thres_target = 500

    v_leak_alpha = 0.8
    v_leak_target = 400

    tau_syn_alpha = 5e6
    tau_syn_target = 5e-6

    tau_m_alpha = 3e6
    tau_m_target = 10e-6

    # use calibration values as initial condition
    v_reset = np.load("{}/v_reset_{}.npy".format(args.prefix, args.chip))
    v_thres = np.load("{}/v_thres_{}.npy".format(args.prefix, args.chip))
    v_leak = np.load("{}/v_leak_{}.npy".format(args.prefix, args.chip))
    tau_syn_exc = np.load("{}/tau_syn_exc_{}.npy".format(args.prefix, args.chip))
    tau_syn_inh = np.load("{}/tau_syn_inh_{}.npy".format(args.prefix, args.chip))
    tau_m = np.load("{}/tau_m_{}.npy".format(args.prefix, args.chip))
    for i in range(n_iterations):
        # measure 
        parameters = {
                halco.CapMemRowOnCapMemBlock.v_leak: v_leak,
                halco.CapMemRowOnCapMemBlock.i_bias_leak: tau_m,
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_res: tau_syn_exc,
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_res: tau_syn_inh,
                halco.CapMemRowOnCapMemBlock.v_threshold: v_thres,
                halco.CapMemRowOnCapMemBlock.v_reset: v_reset
                }
        blackbox.set_neuron_cells(parameters)

        values = measure.measure_tau_syn(blackbox, "exc_synin")
        logger.info("{}: tau_syn_exc={}+/-{}".format(i,
                                                     values.mean(),
                                                     values.std()))
        # update parameter
        tau_syn_exc -= np.clip((tau_syn_alpha * (tau_syn_target - values)).astype(np.int), -10, 10)
        tau_syn_exc = np.clip(tau_syn_exc, 0, 1022)

        values = measure.measure_tau_syn(blackbox, "inh_synin")
        logger.info("{}: tau_syn_inh={}+/-{}".format(i,
                                                     values.mean(),
                                                     values.std()))
        # update parameter
        tau_syn_inh -= np.clip((tau_syn_alpha * (tau_syn_target - values)).astype(np.int), -10, 10)
        tau_syn_inh = np.clip(tau_syn_inh, 0, 1022)

        values = measure.measure_tau_m(blackbox)
        logger.info("{}: tau_m={}+/-{}".format(i, values.mean(), values.std()))

        # update parameter
        tau_m -= np.clip((tau_m_alpha * (tau_m_target - values)).astype(np.int), -10, 10)
        tau_m = np.clip(tau_m, 0, 1022)

        values = measure.measure_v_leak(blackbox, True)
        logger.info("{}: v_leak={}+/-{}".format(i, values.mean(), values.std()))

        # update parameter
        v_leak += np.clip((v_leak_alpha * (v_leak_target - values)).astype(np.int), -10, 10)
        v_leak = np.clip(v_leak, 0, 1022)

        values = measure.measure_v_thres(blackbox)
        logger.info("{}: v_thres={}+/-{}".format(i,
                                                 values.mean(),
                                                 values.std()))

        # update parameter
        v_thres += np.clip((v_thres_alpha * (v_thres_target - values)).astype(np.int), -10, 10)
        v_thres = np.clip(v_thres, 0, 1022)

        values = measure.measure_v_reset(blackbox)
        logger.info("{}: v_reset={}+/-{}".format(i,
                                                 values.mean(),
                                                 values.std()))

        # update parameter
        v_reset += np.clip((v_reset_alpha * (v_reset_target - values)).astype(np.int), -10, 10)
        v_reset = np.clip(v_reset, 0, 1022)

    return v_reset, v_thres, v_leak, tau_syn_exc, tau_syn_inh, tau_m

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", type=int, default=65)
    parser.add_argument("--prefix", type=str, default="data")
    args = parser.parse_args()
    
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')

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
                pass

        # simultaneously calibrate synaptic time constants due to cross talk as in
        # most cases similar values for exc and inh time constant are required
        tau_syn_exc, tau_syn_inh = calibrate_tau_syn(blackbox)
        np.save("{}/tau_syn_exc_{}.npy".format(args.prefix, args.chip), tau_syn_exc)
        np.save("{}/tau_syn_inh_{}.npy".format(args.prefix, args.chip), tau_syn_inh)

        # tau_m = calibrate_tau_m(blackbox)
        # np.save("{}/tau_m_{}.npy".format(args.prefix, args.chip), tau_m)

        # calibrate leak potential roughly to desired value
        # v_leak = calibrate_v_leak(blackbox, enable_synin=False)
        # np.save("{}/v_leak_{}.npy".format(args.prefix, args.chip), v_leak)

        # calibrate synaptic inputs, there is no need to calibrate exc and inh at
        # same time as they could compensate for each other
        # v_syn_exc = calibrate_v_syn_exc(blackbox)
        # np.save("{}/v_syn_exc_{}.npy".format(args.prefix, args.chip), v_syn_exc)
        # v_syn_inh = calibrate_v_syn_inh(blackbox)
        # np.save("{}/v_syn_inh_{}.npy".format(args.prefix, args.chip), v_syn_inh)

        # post calibrate leak potential as calibration of synaptic inputs could have
        # disrupted the previous calibration (cross-talk, calibration imperfection)
        # v_leak = calibrate_v_leak(blackbox, enable_synin=True)
        # np.save("{}/v_leak_{}.npy".format(args.prefix, args.chip), v_leak)

        # v_thres = calibrate_v_thres(blackbox)
        # np.save("{}/v_thres_{}.npy".format(args.prefix, args.chip), v_thres)

        # v_reset = calibrate_v_reset(blackbox)
        # np.save("{}/v_reset_{}.npy".format(args.prefix, args.chip), v_reset)

        # do post  calibration by simultaneously calibrating all LIF parameters
        # v_reset, v_thres, v_leak, tau_syn_exc, tau_syn_inh, tau_m = calibrate_post(blackbox, args)
        # np.save("{}/tau_syn_exc_{}.npy".format(args.prefix, args.chip), tau_syn_exc)
        # np.save("{}/tau_syn_inh_{}.npy".format(args.prefix, args.chip), tau_syn_inh)
        # np.save("{}/tau_m_{}.npy".format(args.prefix, args.chip), tau_m)
        # np.save("{}/v_leak_{}.npy".format(args.prefix, args.chip), v_leak)
        # np.save("{}/v_leak_{}.npy".format(args.prefix, args.chip), v_leak)
        # np.save("{}/v_thres_{}.npy".format(args.prefix, args.chip), v_thres)
        # np.save("{}/v_reset_{}.npy".format(args.prefix, args.chip), v_reset)
