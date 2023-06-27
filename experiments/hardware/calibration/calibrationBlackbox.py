# -*- coding: utf-8 -*-
import numpy as np
import pylogging
from time import time as get_time

import pystadls_vx as stadls
import pyhaldls_vx as haldls
import pyfisch_vx as fisch
import pyhalco_hicann_dls_vx as halco
import pylola_vx as lola

import gonzales

class HXBlackbox:
    def __init__(self,
                 connection,
                 n_neurons,
                 inhibitory_mask=np.zeros(512, dtype=np.int)):
        """
        Initialize blackbox. Each blackbox subclass can define own parameters.
        """
        assert n_neurons <= 256

        self.logger = pylogging.get("HXBlackbox")

        self.connection = connection

        self.n_neurons = n_neurons
        self.inhibitory_mask = inhibitory_mask

        self.vleak_calibration = np.load("calibration/vleak_calibration.npy")
        self.tau_m_calibration = np.load("calibration/tau_m_calibration.npy")
        self.vsyn_exc_calibration = np.load("calibration/vsyn_exc_calibration.npy")
        self.vsyn_inh_calibration = np.load("calibration/vsyn_inh_calibration.npy")
        self.tau_syn_exc_calibration = np.load("calibration/tau_syn_exc_calibration.npy", allow_pickle=True)
        self.tau_syn_inh_calibration = np.load("calibration/tau_syn_inh_calibration.npy", allow_pickle=True)

    def get_connection(self):
        # TODO: do we want this?
        return self.connection

    def initialize(self):
        """
        Reset chip, configure PLL and high speed connection.
        """
        self.logger.info("initialize")

        init_builder, _ = stadls.ExperimentInit().generate()

        # zero all capem cells
        capmem_block_default = haldls.CapMemBlock()
        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            init_builder.write(block, capmem_block_default, haldls.Backend.OmnibusChip)

        # disable spike recording in FPGA
        event_config = haldls.EventRecordingConfig()
        event_config.enable_event_recording = False
        init_builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        stadls.run(self.connection, init_builder.done())

    def configure_capmem(self, config_builder):
        """
        Configure reference generators and capmem.
        """
        self.logger.info("configure_capmem")

        reference_config = haldls.ReferenceGeneratorConfig()
        reference_config.enable_internal_reference = True
        reference_config.enable_reference_output = False
        reference_config.capmem_amplifier = 60
        reference_config.capmem_slope = 5
        reference_config.reference_control = 10
        reference_config.resistor_control = 40

        config_builder.write(halco.ReferenceGeneratorConfigOnDLS(0), reference_config)

        # configure capmem
        capmem_config = haldls.CapMemBlockConfig()
        capmem_config.pulse_a = 11
        capmem_config.pulse_b = 15
        capmem_config.sub_counter = 16
        capmem_config.pause_counter = 8096
        capmem_config.enable_capmem = True

        for cm in halco.iter_all(halco.CapMemBlockConfigOnDLS):
            config_builder.write(cm, capmem_config, haldls.Backend.OmnibusChip)

    def configure_synapses(self, config_builder):
        """
        Configure routing crossbar, PADI bus, synapse drivers, and parts of synapse array.
        """
        self.logger.info("configure_synapses")

        # enintbias
        s = haldls.SynapseBiasSelection()
        for i in halco.iter_all(halco.CapMemBlockOnDLS):
            s.enable_internal_dac_bias[i] = True
            s.enable_internal_output_bias[i] = True
            s.enable_internal_ramp_bias[i] = True
            s.enable_internal_store_bias[i] = True
        config_builder.write(halco.SynapseBiasSelectionOnDLS(), s)

        # synapse array
        correlation_switch_quad = haldls.ColumnCorrelationQuad()
        switch = correlation_switch_quad.ColumnCorrelationSwitch()
        switch.enable_internal_causal = True
        switch.enable_internal_acausal = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            correlation_switch_quad.set_switch(switch_coord, switch)

        for sq in halco.iter_all(halco.ColumnCorrelationQuadOnDLS):
            config_builder.write(sq, correlation_switch_quad, haldls.Backend.OmnibusChip)

        current_switch_quad = haldls.ColumnCurrentQuad()
        switch = current_switch_quad.ColumnCurrentSwitch()
        switch.enable_synaptic_current_excitatory = True
        switch.enable_synaptic_current_inhibitory = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            current_switch_quad.set_switch(switch_coord, switch)

        for sq in halco.iter_all(halco.ColumnCurrentQuadOnDLS):
            config_builder.write(sq, current_switch_quad, haldls.Backend.OmnibusChip)

        # set synapse capmem cells
        synapse_params = {
                halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 1000,
                halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: 0,
                halco.CapMemCellOnCapMemBlock.syn_i_bias_store: 0,
                halco.CapMemCellOnCapMemBlock.syn_i_bias_corout: 0,
                }

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in synapse_params.items():
                config_builder.write(
                        halco.CapMemCellOnDLS(k, block), haldls.CapMemCell(v),
                        haldls.Backend.OmnibusChip
                        )

        # configure synapse SRAM controller
        common_synram_config = haldls.CommonSynramConfig()
        for synram in halco.iter_all(halco.CommonSynramConfigOnDLS):
            config_builder.write(synram, common_synram_config, haldls.Backend.OmnibusChip)

    def configure_routing(self, config_builder):
        self.logger.info("configure_routing")

        # configure PADI bus
        padi_config = haldls.CommonPADIBusConfig()
        for p in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_config.enable_spl1[p] = True
            padi_config.dacen_pulse_extension[p] = 0

        for p in halco.iter_all(halco.CommonPADIBusConfigOnDLS):
            config_builder.write(p, padi_config, haldls.Backend.OmnibusChip)

        active_crossbar_node = haldls.CrossbarNode()
        active_crossbar_node.mask = 0
        active_crossbar_node.target = 0

        silent_crossbar_node = haldls.CrossbarNode()
        silent_crossbar_node.mask = 0
        silent_crossbar_node.target = 2**14 - 1

        # activate recurrent connections within top half
        for i in range(8):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(i % 4),
                        halco.CrossbarInputOnDLS(i)
                        ),
                    active_crossbar_node
                    )

        # clear all inputs
        for o in range(8):
            for i in range(8, 12):
                config_builder.write(
                        halco.CrossbarNodeOnDLS(
                            halco.CrossbarOutputOnDLS(o),
                            halco.CrossbarInputOnDLS(i)
                            ),
                        silent_crossbar_node
                        )

        # enable loopback
        for i in range(4):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(8 + i),
                        halco.CrossbarInputOnDLS(8 + i)
                        ),
                    active_crossbar_node
                    )

        # activate background spike loopback
        for i in range(8):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(8 + (i % 4)),
                        halco.CrossbarInputOnDLS(12 + i)
                        ),
                    active_crossbar_node
                    )

        # enable input from L2 to top and bottom half
        for o in range(8):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(o),
                        halco.CrossbarInputOnDLS(8 + (o % 4))
                        ),
                    active_crossbar_node
                    )

        # configure synapse drivers
        synapse_driver_default = haldls.SynapseDriverConfig()
        for syndrv in halco.iter_all(halco.SynapseDriverOnDLS):
            config_builder.write(syndrv, synapse_driver_default, haldls.Backend.OmnibusChip)

        # enable synapse drivers
        synapse_driver = haldls.SynapseDriverConfig()
        synapse_driver.enable_receiver = True
        synapse_driver.row_address_compare_mask = 0b00000
        synapse_driver.enable_address_out = True

        for d in halco.iter_all(halco.SynapseDriverOnDLS):
            synapse_driver.row_mode_top = getattr(
                    haldls.SynapseDriverConfig.RowMode,
                    "inhibitory" if self.inhibitory_mask[2 * int(d.toEnum()) + 1] else "excitatory"
                    )
            synapse_driver.row_mode_bottom = getattr(
                    haldls.SynapseDriverConfig.RowMode,
                    "inhibitory" if self.inhibitory_mask[2 * int(d.toEnum()) + 0] else "excitatory"
                    )
            config_builder.write(d, synapse_driver, haldls.Backend.OmnibusChip)

    def configure_neuron_backends(self, config_builder):
        """
        Configure neuron backend.
        """
        self.logger.info("configure_neuron_backends")

        # common neuron backend
        common_backend_config = haldls.CommonNeuronBackendConfig()
        common_backend_config.enable_event_registers = True
        common_backend_config.enable_clocks = True
        common_backend_config.clock_scale_slow = 4
        common_backend_config.clock_scale_fast = 4
        for block in halco.iter_all(halco.NeuronEventOutputOnNeuronBackendBlock):
            common_backend_config.set_sample_positive_edge(block, True)

        for backend_block in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            config_builder.write(backend_block, common_backend_config, haldls.Backend.OmnibusChip)

        # neuron backends
        n = 0
        for backend_block in halco.iter_all(halco.NeuronBackendConfigBlockOnDLS):
            for backend in halco.iter_all(halco.NeuronBackendConfigOnNeuronBackendConfigBlock):
                backend_config = haldls.NeuronBackendConfig()

                if backend < 128:
                    backend_config.address_out = int(backend % 32 + (32 if n > 127 else 0)) | ((int(backend.toEnum()) >= 128) << 7)
                    backend_config.enable_spike_out = n < self.n_neurons
                    n += 1
                else:
                    backend_config.enable_spike_out = False

                backend_config.refractory_time = 80
                backend_config.reset_holdoff = 15

                config_builder.write(
                        halco.NeuronBackendConfigOnDLS(backend, backend_block),
                        backend_config,
                        haldls.Backend.OmnibusChip
                        )

    def configure_neurons(self, config_builder, record_neuron=None):
        """
        Configure neurons.
        """
        self.logger.info("configure_neurons")

        # setup neurons
        default_neuron_config = haldls.NeuronConfig()
        for neuron in halco.iter_all(halco.NeuronConfigOnDLS):
            config_builder.write(neuron, default_neuron_config, haldls.Backend.OmnibusChip)

        neuron_config = haldls.NeuronConfig()
        neuron_config.enable_synaptic_input_excitatory = True
        neuron_config.enable_synaptic_input_inhibitory = True

        for neuron in halco.iter_all(halco.NeuronConfigOnDLS):
            if int(neuron.toEnum()) < 256:
                neuron_config.connect_bottom = True
                neuron_config.enable_reset_multiplication = True
                neuron_config.enable_leak_multiplication = False
                neuron_config.membrane_capacitor_size = 63
                neuron_config.enable_threshold_comparator = True
                neuron_config.enable_fire = True
            else:
                neuron_config.connect_bottom = False
                neuron_config.enable_reset_multiplication = True
                neuron_config.enable_leak_multiplication = False
                neuron_config.membrane_capacitor_size = 0
                neuron_config.enable_threshold_comparator = False
                neuron_config.enable_fire = False

            if int(neuron.toEnum()) == record_neuron:
                neuron_config.enable_readout = True
                neuron_config.readout_source = neuron_config.ReadoutSource.membrane
            else:
                neuron_config.enable_readout = False
            neuron_config.enable_readout_amplifier = True

            config_builder.write(neuron, neuron_config, haldls.Backend.OmnibusChip)

    def configure_neuron_cells(self, config_builder):
        self.logger.info("configure_neuron_cells")

        # set capmem cells
        neuron_cells = {
                halco.CapMemRowOnCapMemBlock.v_leak: self.vleak_calibration,
                halco.CapMemRowOnCapMemBlock.v_syn_exc: self.vsyn_exc_calibration,
                halco.CapMemRowOnCapMemBlock.v_syn_inh: self.vsyn_inh_calibration,
                halco.CapMemRowOnCapMemBlock.v_leak_adapt: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_threshold: np.random.randint(340, 360, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_reset: np.random.randint(450, 470, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_leak: 0.3*self.tau_m_calibration[:,0],
                halco.CapMemRowOnCapMemBlock.i_bias_reset: np.random.randint(1000, 1020, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_res: self.tau_syn_exc_calibration[:,0],
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: np.random.randint(1000, 1020, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_res: self.tau_syn_inh_calibration[:,0],
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.random.randint(1000, 1020, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_sd: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_res: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_source_follower: np.random.randint(490, 510, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_offset: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_readout: np.full(halco.NeuronConfigOnDLS.size, 750),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_amp: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_nmda: np.zeros(halco.NeuronConfigOnDLS.size)
                }

        neuron_cells[halco.CapMemRowOnCapMemBlock.i_bias_leak][256:] = 0

        for row, values in neuron_cells.items():
            for block in halco.iter_all(halco.CapMemBlockOnDLS):
                for col in halco.iter_all(halco.CapMemColumnOnCapMemBlock):
                    if int(col) > 127:
                        continue
                    config_builder.write(
                            halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock(col, row), block),
                            haldls.CapMemCell(int(values[block.value() * 128 + col.value()])),
                            haldls.Backend.OmnibusChip
                            )

        # global cells
        neuron_params = {
                halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_exc: 1008,
                halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_inh: 1009,
                halco.CapMemCellOnCapMemBlock.neuron_i_bias_threshold_comparator: 1001,
                }

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in neuron_params.items():
                config_builder.write(halco.CapMemCellOnDLS(k, block), haldls.CapMemCell(v),
                        haldls.Backend.OmnibusChip
                        )

    def configure_readout(self, config_builder):
        self.logger.info("configure_readout")
        
        madc_config = haldls.MADCConfig()
        
        madc_config.active_mux_initially_selected_input = True
        madc_config.active_mux_input_select_length = 0
        madc_config.sample_duration_adjust = 5
        madc_config.sar_reset_wait = 3
        madc_config.sar_reset_length = 0
        madc_config.powerup_wait_value = 96
        madc_config.conversion_cycles_offset = 12
        madc_config.calibration_wait_value = 0
        madc_config.number_of_samples = 40000
        madc_config.preamp_gain_capacitor_size = 31
        madc_config.madc_clock_scale_value = 0
        
        config_builder.write(halco.MADCConfigOnDLS(), madc_config) 

        # set capmem cells
        readout_params = {
                halco.CapMemCellOnDLS.readout_out_amp_i_bias_0: 1000,
                halco.CapMemCellOnDLS.readout_out_amp_i_bias_1: 1000,
                halco.CapMemCellOnDLS.readout_pseudo_diff_buffer_bias: 1000,
                halco.CapMemCellOnDLS.readout_ac_mux_i_bias: 500,
                halco.CapMemCellOnDLS.readout_madc_in_500na: 500,
                halco.CapMemCellOnDLS.readout_sc_amp_i_bias: 500,
                halco.CapMemCellOnDLS.readout_sc_amp_v_ref: 400,
                halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref: 400,
                halco.CapMemCellOnDLS.readout_iconv_test_voltage: 400,
                halco.CapMemCellOnDLS.readout_iconv_sc_amp_v_ref: 400,
                }

        for k, v in readout_params.items():
            config_builder.write(k, haldls.CapMemCell(v), haldls.Backend.OmnibusChip)

    def configure_cadc(self, config_builder):
        self.logger.info("configure_cadc")

        cadc_config = haldls.CADCConfig()
        cadc_config.enable = True
        cadc_config.reset_wait = 150
        cadc_config.dead_time = 200

        for c in halco.iter_all(halco.CADCConfigOnDLS):
            config_builder.write(c, cadc_config, haldls.Backend.OmnibusChip)

        # set capmem cells
        cadc_params = {
                halco.CapMemCellOnCapMemBlock.cadc_v_ramp_offset: 150,
                halco.CapMemCellOnCapMemBlock.cadc_v_bias_ramp_buf: 400,
                halco.CapMemCellOnCapMemBlock.cadc_i_ramp_slope: 211,
                halco.CapMemCellOnCapMemBlock.cadc_i_bias_comp: 1006,
                halco.CapMemCellOnCapMemBlock.cadc_i_bias_vreset_buf: 1015,
                }

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in cadc_params.items():
                config_builder.write(
                        halco.CapMemCellOnDLS(k, block), haldls.CapMemCell(v),
                        haldls.Backend.OmnibusChip
                        )

    def configure(self, pre_hook=None, post_hook=None):
        config_builder = stadls.PlaybackProgramBuilder()

        if pre_hook is not None:
            pre_hook(config_builder)

        self.configure_capmem(config_builder)
        self.configure_neuron_backends(config_builder)
        self.configure_neurons(config_builder)
        self.configure_neuron_cells(config_builder)
        self.configure_readout(config_builder)
        self.configure_cadc(config_builder)
        self.configure_synapses(config_builder)
        self.configure_routing(config_builder)

        if post_hook is not None:
            post_hook(config_builder)

        config_builder.write(halco.TimerOnDLS(), haldls.Timer())
        config_builder.wait_until(halco.TimerOnDLS(), 10000*fisch.fpga_clock_cycles_per_us)

        stadls.run(self.connection, config_builder.done())

    def set_readout(self, record_neuron):
        """
        Configure readout chain to route analog output of given neuron to MADC.
        """

        builder = stadls.PlaybackProgramBuilder()
        neuron_readout_line = 2*(int(record_neuron // 128) > 1) + 1 \
               - int(record_neuron % 128) % 2

        hemisphere = record_neuron // 128
        is_odd = (record_neuron % 2 + 1) > 0
        is_even = (record_neuron % 2) > 0

        # config = haldls.ReadoutBufferConfigBlock()
        # rbc = config.ReadoutBufferConfig()
        # rbc.neuron_odd[halco.HemisphereOnDLS(hemisphere)] = is_odd
        # rbc.neuron_even[halco.HemisphereOnDLS(hemisphere)] = is_even
        # rbc.enable_buffer = True
        # config.set_buffer(halco.ReadoutBufferConfigOnReadoutBufferConfigBlock(1), rbc)
        # builder.write(halco.ReadoutBufferConfigBlockOnDLS(), config)

        fisch_builder = fisch.PlaybackProgramBuilder()
        madc_config_reg = 0
        madc_config_reg |= (1 << neuron_readout_line) << 9 # mux[0:12] # s0, s1, n0, n1
        madc_config_reg |= 1 << 13 # out_amp_en[0:1]

        madc_base_address = 1 << 19 | 1 << 18
        fisch_builder.write(madc_base_address + 13, fisch.OmnibusChip(madc_config_reg))

        builder.merge_back(fisch_builder)

        self.configure_neurons(builder, record_neuron)

        stadls.run(self.connection, builder.done())

    def set_weights(self, weights):
        """
        This takes *logical* weight matrices, corresponding to the topology
        determined by `configure_routing`.
        """

        builder = stadls.PlaybackProgramBuilder()

        rows = np.arange(halco.SynapseDriverOnDLS.size)
        drivers = rows // 2
        buses = drivers % 4
        rows_on_bus = 2*(rows // 8) + (rows % 2)
        source_neurons = rows_on_bus + 32 * buses + (128 - 32) * (rows >= 128)

        weights_top = weights[:256, :]
        weights_bottom = weights[256:, :]
        
        synapse_matrix = lola.SynapseMatrix()
        synapse_matrix.labels.from_numpy(np.tile(rows_on_bus, (256, 1)).T)
        synapse_matrix.weights.from_numpy(weights_top[source_neurons,:])
        builder.write(halco.SynramOnDLS.top, synapse_matrix, haldls.Backend.OmnibusChip)
        
        synapse_matrix = lola.SynapseMatrix()
        synapse_matrix.labels.from_numpy(np.tile(rows_on_bus, (256, 1)).T)
        synapse_matrix.weights.from_numpy(weights_bottom[source_neurons,:])
        builder.write(halco.SynramOnDLS.bottom, synapse_matrix, haldls.Backend.OmnibusChip)

        stadls.run(self.connection, builder.done())

    def stimulate(self, 
                  duration,
                  input_spikes=None,
                  spike_record=False,
                  madc_record=False,
                  pre_hook=None,
                  post_hook=None):

        builder = stadls.PlaybackProgramBuilder()

        if input_spikes is not None:
            builder.write(halco.TimerOnDLS(), haldls.Timer())
            builder.wait_until(halco.TimerOnDLS(), 5000 * fisch.fpga_clock_cycles_per_us)

        # set up recording -- or not
        if madc_record is True:
            # start MADC
            madc_control = haldls.MADCControl()
            madc_control.wake_up = True
            madc_control.start_recording = True
            madc_control.enable_continuous_sampling = False
            madc_control.enable_power_down_after_sampling = True
            madc_control.enable_pre_amplifier = True
            builder.write(halco.MADCControlOnDLS(), madc_control, haldls.Backend.OmnibusChip)

            # stop recording after the configured number of samples
            madc_control.wake_up = False
            builder.write(halco.MADCControlOnDLS(), madc_control, haldls.Backend.OmnibusChip)

        builder.write(halco.TimerOnDLS(), haldls.Timer())
        if pre_hook is not None:
            pre_hook(builder)

        # sync time
        builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
        builder.write(halco.TimerOnDLS(), haldls.Timer())
        builder.wait_until(halco.TimerOnDLS(), 100)

        if spike_record is True:
            event_config = haldls.EventRecordingConfig()
            event_config.enable_event_recording = True
            builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        if input_spikes is not None:
            for time, label in input_spikes:
                bus_to_fire = (int(label) // 32) % 4
                event_address = int(label) % 32 + 32*(int(label) // 128)

                builder.wait_until(
                        halco.TimerOnDLS(),
                        int(time * 1e6 * fisch.fpga_clock_cycles_per_us))

                spike_label = haldls.SpikeLabel()
                spike_label.neuron_label = event_address
                spike_label.spl1_address = bus_to_fire
                builder.write(
                        halco.SpikePack1ToChipOnDLS(),
                        haldls.SpikePack1ToChip([spike_label])
                        )

        builder.wait_until(
                halco.TimerOnDLS(),
                int(duration * 1e6 * fisch.fpga_clock_cycles_per_us))

        if spike_record is True:
            event_config = haldls.EventRecordingConfig()
            event_config.enable_event_recording = False
            builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        if post_hook is not None:
            post_hook(builder)
        
        tickets = list()
        for counter_read in halco.iter_all(halco.SpikeCounterReadOnDLS):
            tickets.append(builder.read(counter_read))


        program = builder.done()
        stadls.run(self.connection, program)

        rates = np.zeros(len(tickets))
        overflows = np.zeros(len(tickets), dtype=np.bool)
        for i, t in enumerate(tickets):
            rates[i] = int(t.get().count)
            overflows[i] = int(t.get().overflow)

        rates[overflows] = 2**10
        rates = rates[np.concatenate((np.arange(0, 128), np.arange(256, 384)))]
        rates /= duration

        if spike_record is True:
            self.logger.info("Found {} spikes".format(len(program.spikes)))

            spikes = program.spikes.to_numpy()
            times = spikes["chip_time"] / float(fisch.fpga_clock_cycles_per_us) * 1e-6
            labels = spikes["label"]

            # construct masks
            is_input = ((labels >> 13) & 0b1)
            input_mask = (is_input == 1)
            network_mask = (is_input == 0)

            # reconstruct input spike labels
            is_bottom = ((labels[input_mask] >> 7) & 0b1)
            generator_id = ((labels[input_mask] >> 11) & 0b11) + 4 * is_bottom
            neuron_address = (labels[input_mask] & 0b111111)
            
            input_spikes = np.zeros((input_mask.sum(), 2))
            input_spikes[:, 0] = times[input_mask]
            input_spikes[:, 1] = 64 * generator_id + neuron_address
            
            # reconstruct network spike labels
            neuron_labels = labels[network_mask] & 2**14 - 1
            bus = labels[network_mask] >> 14
            neuron_blocks = neuron_labels // 256

            network_spikes = np.zeros((network_mask.sum(), 2))
            network_spikes[:, 0] = times[network_mask]
            network_spikes[:, 1] = neuron_labels - 256 * neuron_blocks + bus * 32 + (128 - 32) * ((neuron_labels % 256).astype(np.int) >= 32)
        else:
            input_spikes = None
            network_spikes = None

        if madc_record is True:
            madc_samples = program.madc_samples.to_numpy()
            samples = np.zeros((madc_samples.size, 2), dtype=np.float)
            samples[:, 0] = madc_samples["value"]
            samples[:, 1] = madc_samples["chip_time"]

            samples[:,1] /= (float(fisch.fpga_clock_cycles_per_us) * 1e6)
            samples = samples[10:, :]
        else:
            samples = None

        return input_spikes, network_spikes, rates, samples


class CalibrationBlackbox(HXBlackbox):
    _neuron_size = 1
    _input_addresses_top = np.arange(256) % 32
    _input_buses_top = np.arange(256) // 32

    _neuron_addresses_top = (np.arange(256) % 32) + 32 * (np.arange(256) // 128)
    _neuron_buses_top = (np.arange(256) // 32) % 4

    _input_addresses_bottom = _input_addresses_top
    _input_buses_bottom = _input_buses_top
    _neuron_addresses_bottom = _neuron_addresses_top
    _neuron_buses_bottom = _neuron_buses_top

    def __init__(self, connection):
        self.logger = pylogging.get(self.__class__.__name__)

        self._enable_spiking = False

        self.connection = connection

    def configure_routing(self, config_builder):
        self.logger.info("configure_routing")

        # configure PADI bus
        padi_config = haldls.CommonPADIBusConfig()
        for p in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_config.enable_spl1[p] = True
            padi_config.dacen_pulse_extension[p] = 0

        for p in halco.iter_all(halco.CommonPADIBusConfigOnDLS):
            config_builder.write(p, padi_config, haldls.Backend.OmnibusChip)

        active_crossbar_node = haldls.CrossbarNode()
        active_crossbar_node.mask = 0
        active_crossbar_node.target = 0

        silent_crossbar_node = haldls.CrossbarNode()
        silent_crossbar_node.mask = 0
        silent_crossbar_node.target = 2**14 - 1

        for node in halco.iter_all(halco.CrossbarNodeOnDLS):
            config_builder.write(node, silent_crossbar_node)

        # activate neuron spike output
        for i in range(8):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(8 + (i % 4)),
                        halco.CrossbarInputOnDLS(i)
                        ),
                    active_crossbar_node
                    )

        # enable input from L2 to top and bottom
        for o in range(0, 8):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(o),
                        halco.CrossbarInputOnDLS(8 + (o % 4))
                        ),
                    active_crossbar_node
                    )

        # configure synapse drivers
        synapse_driver_default = haldls.SynapseDriverConfig()
        for syndrv in halco.iter_all(halco.SynapseDriverOnDLS):
            config_builder.write(syndrv, synapse_driver_default, haldls.Backend.OmnibusChip)

        # enable synapse drivers
        synapse_driver = haldls.SynapseDriverConfig()
        synapse_driver.enable_receiver = True
        synapse_driver.row_address_compare_mask = 0b00000
        synapse_driver.enable_address_out = True
        synapse_driver.row_mode_top = haldls.SynapseDriverConfig.RowMode.inhibitory
        synapse_driver.row_mode_bottom = haldls.SynapseDriverConfig.RowMode.excitatory

        for d in halco.iter_all(halco.SynapseDriverOnDLS):
            config_builder.write(d, synapse_driver, haldls.Backend.OmnibusChip)

    def configure_neuron_cells(self, config_builder):
        self.logger.info("configure_neuron_cells")

        # set capmem cells
        neuron_cells = {
                halco.CapMemRowOnCapMemBlock.v_leak: np.random.randint(900, 940, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_syn_exc: np.random.randint(600, 650, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_syn_inh: np.random.randint(600, 650, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_leak_adapt: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_threshold: np.random.randint(340, 360, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_reset: np.random.randint(450, 470, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_leak:  np.random.randint(600, 650, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_reset: np.random.randint(1000, 1020, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_res: np.random.randint(400, 450, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: np.random.randint(1000, 1020, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_res: np.random.randint(400, 450, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.random.randint(1000, 1020, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_sd: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_res: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_source_follower: np.random.randint(490, 510, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_offset: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_readout: np.full(halco.NeuronConfigOnDLS.size, 1000),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_amp: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_nmda: np.zeros(halco.NeuronConfigOnDLS.size)
                }

        for row, values in neuron_cells.items():
            for block in halco.iter_all(halco.CapMemBlockOnDLS):
                for col in halco.iter_all(halco.CapMemColumnOnCapMemBlock):
                    if int(col) > 127:
                        continue
                    config_builder.write(
                            halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock(col, row), block),
                            haldls.CapMemCell(int(values[block.value() * 128 + col.value()])),
                            haldls.Backend.OmnibusChip
                            )

        # global cells
        neuron_params = {
                halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_exc: 1008,
                halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_inh: 1009,
                halco.CapMemCellOnCapMemBlock.neuron_i_bias_threshold_comparator: 1001,
                }

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in neuron_params.items():
                config_builder.write(halco.CapMemCellOnDLS(k, block), haldls.CapMemCell(v), haldls.Backend.OmnibusChip)

    def set_neuron_cells(self, neuron_cells):
        self.logger.info("set_neuron_cells")

        config_builder = stadls.PlaybackProgramBuilder()

        for row, values in neuron_cells.items():
            for block in halco.iter_all(halco.CapMemBlockOnDLS):
                for col in halco.iter_all(halco.CapMemColumnOnCapMemBlock):
                    if int(col) > 127:
                        continue
                    config_builder.write(
                            halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock(col, row), block),
                            haldls.CapMemCell(int(values[block.value() * 128 + col.value()])),
                            haldls.Backend.OmnibusChip
                            )

        stadls.run(self.connection, config_builder.done())

    def configure_neuron_backends(self, config_builder):
        """
        Configure neuron backend.
        """
        self.logger.info("configure_neuron_backends")

        # common neuron backend
        common_backend_config = haldls.CommonNeuronBackendConfig()
        common_backend_config.enable_event_registers = True
        common_backend_config.enable_clocks = True
        common_backend_config.clock_scale_slow = 4
        common_backend_config.clock_scale_fast = 4
        for block in halco.iter_all(halco.NeuronEventOutputOnNeuronBackendBlock):
            common_backend_config.set_sample_positive_edge(block, True)

        for backend_block in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            config_builder.write(backend_block, common_backend_config, haldls.Backend.OmnibusChip)

        slave_config_proto = haldls.NeuronBackendConfig()
        slave_config_proto.enable_spike_out = False

        master_config_proto = haldls.NeuronBackendConfig()
        master_config_proto.enable_spike_out = True
        master_config_proto.refractory_time = 80
        master_config_proto.reset_holdoff = 15

        for backend_block in halco.iter_all(halco.NeuronBackendConfigBlockOnDLS):
            is_right = backend_block.toEnum() == 1
            for backend in halco.iter_all(halco.NeuronBackendConfigOnNeuronBackendConfigBlock):
                is_bottom = backend.toEnum() >= 128
                backend_index_on_quadrant = int(backend.toEnum() % 128)

                backend_config = haldls.NeuronBackendConfig(master_config_proto)
                if not is_bottom:
                    address = self._neuron_addresses_top[backend_index_on_quadrant + 128 * is_right]
                else:
                    address = self._neuron_addresses_bottom[backend_index_on_quadrant + 128 * is_right]

                # annotate if neuron is top or bottom
                address |= (is_bottom << 7)
                backend_config.address_out = int(address)

                config_builder.write(
                        halco.NeuronBackendConfigOnDLS(backend, backend_block),
                        backend_config,
                        haldls.Backend.OmnibusChip
                        )

    def configure_neurons(self, config_builder, record_neuron=None, record_target="membrane"):
        """
        Configure neurons.
        """
        self.logger.debug("configure_neurons")

        default_neuron_config = haldls.NeuronConfig()
        default_neuron_config.enable_synaptic_input_excitatory = True
        default_neuron_config.enable_synaptic_input_inhibitory = True
        default_neuron_config.enable_reset_multiplication = True
        default_neuron_config.enable_leak_multiplication = False
        default_neuron_config.membrane_capacitor_size = 63
        default_neuron_config.enable_threshold_comparator = self._enable_spiking
        default_neuron_config.enable_fire = self._enable_spiking
        default_neuron_config.enable_readout_amplifier = True
        default_neuron_config.enable_readout = False

        for neuron in halco.iter_all(halco.NeuronConfigOnDLS):
            neuron_config = haldls.NeuronConfig(default_neuron_config)
            if int(neuron.toEnum()) == record_neuron:
                neuron_config.enable_readout = True
                neuron_config.enable_readout_amplifier = True
                neuron_config.readout_source = getattr(neuron_config.ReadoutSource, record_target)
            config_builder.write(neuron, neuron_config, haldls.Backend.OmnibusChip)

    def set_enable_spiking(self, enable_spiking):
        self._enable_spiking = enable_spiking

    def set_weights(self, weights_input_hidden=None, weights_hidden_output=None):
        """
        This takes *logical* weight matrices, corresponding to the topology
        determined by `configure_routing`.
        """

        expected_shape = (128*self._neuron_size, 256//self._neuron_size)

        builder = stadls.PlaybackProgramBuilder()

        if weights_input_hidden is not None:
            weights = weights_input_hidden
            n_logical_inputs = 128
            is_signed = True

            expected_shape = (n_logical_inputs, 256 // self._neuron_size)
            assert weights.shape == expected_shape

            rows = np.arange(n_logical_inputs * (1 + is_signed))
            drivers = (rows // 2) % 128
            buses = drivers % 4
            rows_on_buses = ((rows % 256) // (4 + 4*is_signed))
            signs_on_rows = rows % (1 + is_signed)
            wrap_counts = rows // 256

            addresses = rows_on_buses + 32*wrap_counts

            target_synapse = np.arange(4)

            synapse_lookup = np.zeros(rows.size, dtype=np.int)
            for target_synapse in np.arange(synapse_lookup.size):
                synapse_lookup[target_synapse] = np.where(
                        (self._input_buses_top == buses[target_synapse])
                      & (self._input_addresses_top == addresses[target_synapse])
                      )[0][0]

            weights_exc = np.clip(weights[synapse_lookup[::2]], 0, 63)
            weights_inh = np.clip(weights[synapse_lookup[::2]], -63, 0) * (-1)

            weights_unsigned = np.hstack([weights_exc, weights_inh]).reshape(-1, weights_exc.shape[1])
            weights_unsigned = weights_unsigned.T.reshape(256, 256).T

            addresses = np.tile(addresses, (256, 1)).T
            addresses = addresses.T.reshape(256, 256).T

            synapse_matrix = lola.SynapseMatrix()
            synapse_matrix.labels.from_numpy(addresses)
            synapse_matrix.weights.from_numpy(weights_unsigned)
            builder.write(halco.SynramOnDLS.top, synapse_matrix, haldls.Backend.OmnibusChip)

        if weights_hidden_output is not None:
            """
            weights = weights_hidden_output
            n_logical_inputs = 128
            is_signed = True

            expected_shape = (n_logical_inputs, 256)
            assert weights.shape == expected_shape

            rows = np.arange(n_logical_inputs * (1 + is_signed))
            drivers = (rows // 2) % 128
            buses = drivers % 4
            rows_on_buses = ((rows % 256) // (4 + 4*is_signed))
            signs_on_rows = rows % (1 + is_signed)
            wrap_counts = rows // 256

            addresses = rows_on_buses + 32*wrap_counts

            target_synapse = np.arange(4)

            synapse_lookup = np.zeros(rows.size, dtype=np.int)
            for target_synapse in np.arange(synapse_lookup.size):
                synapse_lookup[target_synapse] = np.where(
                        (self._input_buses_bottom == buses[target_synapse])
                      & (self._input_addresses_bottom == addresses[target_synapse])
                      )[0][0] // 2

            weights_exc = np.clip(weights[synapse_lookup[::2]], 0, 63)
            weights_inh = np.clip(weights[synapse_lookup[::2]], -63, 0) * (-1)

            number_expected_rows = 256 // (1 + is_signed) * self._neuron_size
            weights_exc = np.pad(weights_exc, ((0, number_expected_rows - weights_exc.shape[0]), (0, 0)))
            weights_inh = np.pad(weights_inh, ((0, number_expected_rows - weights_inh.shape[0]), (0, 0)))

            weights_unsigned = np.hstack([weights_exc, weights_inh]).reshape(-1, weights_exc.shape[1])
            weights_unsigned = weights_unsigned.T.reshape(256, 256).T

            addresses = np.tile(addresses, (256, 1)).T
            addresses = np.pad(addresses, ((0, 2*number_expected_rows - addresses.shape[0]), (0, 0)))
            addresses = addresses.T.reshape(256, 256).T

            synapse_matrix = lola.SynapseMatrix()
            synapse_matrix.addresses.from_numpy(addresses)
            print(addresses[:10, :10])
            print(weights_unsigned[:10, :10])
            synapse_matrix.weights.from_numpy(weights_unsigned)
            builder.write(halco.SynramOnDLS.bottom, synapse_matrix, haldls.Backend.OmnibusChip)
            """
            weights = weights_hidden_output
            n_logical_inputs = 128
            is_signed = True

            expected_shape = (n_logical_inputs, 256 // self._neuron_size)
            assert weights.shape == expected_shape

            rows = np.arange(n_logical_inputs * (1 + is_signed))
            drivers = (rows // 2) % 128
            buses = drivers % 4
            rows_on_buses = ((rows % 256) // (4 + 4*is_signed))
            signs_on_rows = rows % (1 + is_signed)
            wrap_counts = rows // 256

            addresses = rows_on_buses + 32*wrap_counts

            target_synapse = np.arange(4)

            synapse_lookup = np.zeros(rows.size, dtype=np.int)
            for target_synapse in np.arange(synapse_lookup.size):
                synapse_lookup[target_synapse] = np.where(
                        (self._input_buses_top == buses[target_synapse])
                      & (self._input_addresses_top == addresses[target_synapse])
                      )[0][0]

            weights_exc = np.clip(weights[synapse_lookup[::2]], 0, 63)
            weights_inh = np.clip(weights[synapse_lookup[::2]], -63, 0) * (-1)

            weights_unsigned = np.hstack([weights_exc, weights_inh]).reshape(-1, weights_exc.shape[1])
            weights_unsigned = weights_unsigned.T.reshape(256, 256).T

            addresses = np.tile(addresses, (256, 1)).T
            addresses = addresses.T.reshape(256, 256).T

            synapse_matrix = lola.SynapseMatrix()
            synapse_matrix.labels.from_numpy(addresses)
            synapse_matrix.weights.from_numpy(weights_unsigned)
            builder.write(halco.SynramOnDLS.bottom, synapse_matrix, haldls.Backend.OmnibusChip)

        stadls.run(self.connection, builder.done())

    def stimulate(self,
                  duration,
                  input_spikes,
                  experiment_builder=None,
                  madc_record=False,
                  pre_hook=None,
                  post_hook=None):

        builder = stadls.PlaybackProgramBuilder()

        if input_spikes is not None:
            builder.write(halco.TimerOnDLS(), haldls.Timer())
            builder.wait_until(halco.TimerOnDLS(), 100 * fisch.fpga_clock_cycles_per_us)

        # set up recording -- or not
        if madc_record is True:
            # start MADC
            madc_control = haldls.MADCControl()
            madc_control.wake_up = True
            madc_control.start_recording = True
            madc_control.enable_continuous_sampling = False
            madc_control.enable_power_down_after_sampling = True
            madc_control.enable_pre_amplifier = True
            builder.write(halco.MADCControlOnDLS(), madc_control, haldls.Backend.OmnibusChip)

            # stop recording after the configured number of samples
            madc_control.wake_up = False
            builder.write(halco.MADCControlOnDLS(), madc_control, haldls.Backend.OmnibusChip)

        builder.write(halco.TimerOnDLS(), haldls.Timer())
        if pre_hook is not None:
            pre_hook(builder)

        # sync time
        builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
        builder.write(halco.TimerOnDLS(), haldls.Timer())
        builder.wait_until(halco.TimerOnDLS(), 100)

        # start FPGA event recording
        event_config = haldls.EventRecordingConfig()
        event_config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        if experiment_builder is not None:
            builder.merge_back(experiment_builder)

        gonzales.generate_spiketrain(
                builder,
                input_spikes[:, 0],
                self._input_addresses_top[input_spikes[:, 1].astype(np.int)],
                self._input_buses_top[input_spikes[:, 1].astype(np.int)]
                )


        builder.wait_until(
                halco.TimerOnDLS(),
                int(duration * 1e6 * fisch.fpga_clock_cycles_per_us))

        # stop FPGA event recording
        event_config = haldls.EventRecordingConfig()
        event_config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        if post_hook is not None:
            post_hook(builder)

        program = builder.done()
        stadls.run(self.connection, program)

        self.logger.debug("Found {} spikes".format(len(program.spikes)))

        spikes = program.spikes.to_numpy()
        times = spikes["chip_time"] / float(fisch.fpga_clock_cycles_per_us) * 1e-6
        labels = spikes["label"]

        # reconstruct network spike labels
        neuron_address = labels & 2**6 - 1
        bus = (labels & (0b11 << 14)) >> 14
        bottom_mask = ((labels & (1 << 7)) >> 7).astype(np.bool)
        top_mask = np.invert(bottom_mask)

        top_spikes = np.zeros((top_mask.sum(), 2))
        # for i in range(neuron_address[top_mask].size):
        #     top_spikes[i, 0] = times[top_mask][i]
        #     top_spikes[i, 1] = np.where(
        #                 (self._neuron_buses_top == bus[i])
        #               & (self._neuron_addresses_top == neuron_address[i])
        #               )[0][0] // self._neuron_size
        spike_resolution_top = np.zeros(
            (self._neuron_addresses_top.max() + 1, self._neuron_buses_top.max() + 1), dtype=np.int)
        for b in np.unique(self._neuron_buses_top):
            bus_mask = (self._neuron_buses_top[::-1] == b)
            spike_resolution_top[self._neuron_addresses_top[::-1][bus_mask], b] = np.arange(256)[::-1][bus_mask]
        top_spikes[:, 0] = times[top_mask]
        top_spikes[:, 1] = spike_resolution_top[neuron_address[top_mask], bus[top_mask]]

        bottom_spikes = np.zeros((bottom_mask.sum(), 2))
        # for i in range(neuron_address[bottom_mask].size):
        #     bottom_spikes[i, 0] = times[bottom_mask][i]
        #     bottom_spikes[i, 1] = np.where(
        #                 (self._neuron_buses_bottom == bus[i])
        #               & (self._neuron_addresses_bottom == neuron_address[i])
        #               )[0][0] // self._neuron_size
        spike_resolution_bottom = np.zeros(
            (self._neuron_addresses_bottom.max() + 1, self._neuron_buses_bottom.max() + 1), dtype=np.int)
        for b in np.unique(self._neuron_buses_bottom):
            bus_mask = (self._neuron_buses_bottom[::-1] == b)
            spike_resolution_bottom[self._neuron_addresses_bottom[::-1][bus_mask], b] = np.arange(256)[::-1][bus_mask]
        bottom_spikes[:, 0] = times[bottom_mask]
        bottom_spikes[:, 1] = spike_resolution_bottom[neuron_address[bottom_mask], bus[bottom_mask]]

        if madc_record is True:
            madc_samples = program.madc_samples.to_numpy()
            samples = np.zeros((madc_samples.size, 2), dtype=np.float)
            samples[:, 0] = madc_samples["value"]
            samples[:, 1] = madc_samples["chip_time"]

            samples[:,1] /= (float(fisch.fpga_clock_cycles_per_us) * 1e6)
            samples = samples[10:, :]
        else:
            samples = None

        return top_spikes, bottom_spikes, samples

    def set_readout(self, record_neuron, readout_target="membrane"):
        """
        Configure readout chain to route analog output of given neuron to MADC.
        """

        builder = stadls.PlaybackProgramBuilder()
        neuron_readout_line = 2*(int(record_neuron // 128) > 1) + 1 \
               - int(record_neuron % 128) % 2

        hemisphere = record_neuron // 128
        is_odd = (record_neuron % 2 + 1) > 0
        is_even = (record_neuron % 2) > 0

        fisch_builder = fisch.PlaybackProgramBuilder()
        madc_config_reg = 0
        madc_config_reg |= (1 << neuron_readout_line) << 9 # mux[0:12] # s0, s1, n0, n1
        madc_config_reg |= 1 << 13 # out_amp_en[0:1]

        madc_base_address = 1 << 19 | 1 << 18
        fisch_builder.write(madc_base_address + 13, fisch.OmnibusChip(madc_config_reg))

        builder.merge_back(fisch_builder)

        self.configure_neurons(builder, record_neuron, readout_target)

        stadls.run(self.connection, builder.done())
