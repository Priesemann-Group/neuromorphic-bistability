# -*- coding: utf-8 -*-
import os
import enum
import pylogging
import numpy as np
import helpers as hp

import gonzales

import pystadls_vx_v1 as stadls
import pyhaldls_vx_v1 as haldls
import pyfisch_vx as fisch
import pyhalco_hicann_dls_vx_v1 as halco
import pylola_vx_v1 as lola


class PPUSignal(enum.Enum):
    RUN = 0
    NONE = 1
    HALT = 2
    CONFIGURE = 3


class HXBlackbox:
    def __init__(self,
                 connection,
                 inhibitory_mask=np.zeros(512, dtype=np.int),
                 enable_loopback=True,
                 use_calibration=True):
        """
        Initialize blackbox. Each blackbox subclass can define own parameters.
        """
        self.logger = pylogging.get("HXBlackbox")

        # self.executor = stadls.PlaybackProgramExecutor()
        # self.executor.connect()
        self.connection = connection

        self.inhibitory_mask = inhibitory_mask
        self.enable_loopback = enable_loopback

        self._prefix = os.path.abspath(os.path.join(__file__,
                                                    os.pardir,
                                                    os.pardir,
                                                    "calibration",
                                                    "data"))


        self.v_leak = np.load(os.path.join(self._prefix, "v_leak_68.npy"))
        self.v_thres = np.load(os.path.join(self._prefix, "v_thres_68.npy"))
        self.v_reset = np.load(os.path.join(self._prefix, "v_reset_68.npy"))
        self.v_syn_exc = np.load(os.path.join(self._prefix, "v_syn_exc_68.npy"))
        self.v_syn_inh = np.load(os.path.join(self._prefix, "v_syn_inh_68.npy"))
        self.tau_m = np.load(os.path.join(self._prefix, "tau_m_68.npy"))
        self.tau_syn_exc = np.load(os.path.join(self._prefix, "tau_syn_exc_68.npy"))
        self.tau_syn_inh = np.load(os.path.join(self._prefix, "tau_syn_inh_68.npy"))

        if not use_calibration:
            self.tau_m = hp.jitter(self.tau_m.mean())
            self.tau_syn_exc = hp.jitter(self.tau_syn_exc.mean())
            self.tau_syn_inh = hp.jitter(self.tau_syn_inh.mean())
    
    def get_connection(self):
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
            init_builder.write(block, capmem_block_default, haldls.Backend.Omnibus)

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

        ref_config = haldls.ReferenceGeneratorConfig()
        ref_config.enable_internal_reference = True
        ref_config.enable_reference_output = False
        ref_config.capmem_amplifier = 60
        ref_config.capmem_slope = 5
        ref_config.reference_control = 10
        ref_config.resistor_control = 40

        config_builder.write(halco.ReferenceGeneratorConfigOnDLS(0), ref_config)

        # configure capmem
        capmem_config = haldls.CapMemBlockConfig()
        capmem_config.pulse_a = 11
        capmem_config.pulse_b = 15
        capmem_config.sub_counter = 16
        capmem_config.pause_counter = 8096
        capmem_config.enable_capmem = True

        for cm in halco.iter_all(halco.CapMemBlockConfigOnDLS):
            config_builder.write(cm, capmem_config, haldls.Backend.Omnibus)

    def configure_synapses(self, config_builder):
        """
        Configure routing crossbar, PADI bus, synapse drivers, and parts of
        synapse array.
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
            config_builder.write(sq, correlation_switch_quad, haldls.Backend.Omnibus)

        current_switch_quad = haldls.ColumnCurrentQuad()
        switch = current_switch_quad.ColumnCurrentSwitch()
        switch.enable_synaptic_current_excitatory = True
        switch.enable_synaptic_current_inhibitory = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            current_switch_quad.set_switch(switch_coord, switch)

        for quad_coord in halco.iter_all(halco.ColumnCurrentQuadOnDLS):
            config_builder.write(quad_coord, current_switch_quad, haldls.Backend.Omnibus)

        # set synapse capmem cells
        synapse_params = {
                halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 1000,
                halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: 0,
                halco.CapMemCellOnCapMemBlock.syn_i_bias_store: 0,
                halco.CapMemCellOnCapMemBlock.syn_i_bias_corout: 0,
                }

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in synapse_params.items():
                config_builder.write(halco.CapMemCellOnDLS(k, block), haldls.CapMemCell(v),
                        haldls.Backend.Omnibus
                        )

        # configure synapse SRAM controller
        common_synram_config = haldls.CommonSynramConfig()
        for synram in halco.iter_all(halco.CommonSynramConfigOnDLS):
            config_builder.write(synram, common_synram_config, haldls.Backend.Omnibus)

    def configure_routing(self, config_builder):
        self.logger.info("configure_routing")

        # configure PADI bus
        padi_config = haldls.CommonPADIBusConfig()
        for p in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_config.enable_spl1[p] = True
            padi_config.dacen_pulse_extension[p] = 0

        for p in halco.iter_all(halco.CommonPADIBusConfigOnDLS):
            config_builder.write(p, padi_config, haldls.Backend.Omnibus)

        active_crossbar_top_node = haldls.CrossbarNode()
        active_crossbar_top_node.mask = 1 << 7
        active_crossbar_top_node.target = 0
        
        active_crossbar_bottom_node = haldls.CrossbarNode()
        active_crossbar_bottom_node.mask = 1 << 7
        active_crossbar_bottom_node.target = 1 << 7
        
        active_crossbar_loopback_node = haldls.CrossbarNode()
        active_crossbar_loopback_node.mask = 0
        active_crossbar_loopback_node.target = 0

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
                    active_crossbar_loopback_node
                    )
        
        # disable recurrent connections within bottom half
        for i in range(8):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS((i % 4) + 4),
                        halco.CrossbarInputOnDLS(i)
                        ),
                    silent_crossbar_node
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
                    active_crossbar_loopback_node if self.enable_loopback else silent_crossbar_node
                    )

        # activate background spike loopback
        for i in range(8):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(8 + (i % 4)),
                        halco.CrossbarInputOnDLS(12 + i)
                        ),
                    active_crossbar_loopback_node if self.enable_loopback else silent_crossbar_node
                    )

        # enable input from L2 to top half
        for o in range(4):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(o),
                        halco.CrossbarInputOnDLS(8 + o)
                        ),
                    active_crossbar_top_node
                    )
        
        # enable input from L2 to bottom half
        for o in range(4):
            config_builder.write(
                    halco.CrossbarNodeOnDLS(
                        halco.CrossbarOutputOnDLS(4 + o),
                        halco.CrossbarInputOnDLS(8 + o)
                        ),
                    active_crossbar_bottom_node
                    )

        # configure synapse drivers
        synapse_driver_default = haldls.SynapseDriverConfig()
        for syndrv in halco.iter_all(halco.SynapseDriverOnDLS):
            config_builder.write(syndrv, synapse_driver_default, haldls.Backend.Omnibus)

    def configure_synapse_drivers(self, config_builder):
        """
        Configure synapse drivers.
        """
#        self.logger.info("configure_synapse_drivers")

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
            config_builder.write(d, synapse_driver, haldls.Backend.Omnibus)

    def configure_neuron_backends(self, config_builder):
        """
        Configure neuron backend.
        """
        self.logger.info("configure_neuron_backends")

        # common neuron backend
        common_backend_config = haldls.CommonNeuronBackendConfig()
        common_backend_config.enable_event_registers = True
        common_backend_config.enable_clocks = True
        common_backend_config.clock_scale_slow = 5
        common_backend_config.clock_scale_fast = 5
        common_backend_config.wait_spike_counter_reset = 113
        for block in halco.iter_all(halco.NeuronEventOutputOnNeuronBackendBlock):
            common_backend_config.set_sample_positive_edge(block, True)

        for backend_block in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            config_builder.write(backend_block, common_backend_config, haldls.Backend.Omnibus)

        # neuron backends
        n = 0
        for backend_block in halco.iter_all(halco.NeuronBackendConfigBlockOnDLS):
            for backend in halco.iter_all(halco.NeuronBackendConfigOnNeuronBackendConfigBlock):
                backend_config = haldls.NeuronBackendConfig()

                if backend < 128:
                    backend_config.address_out = int(backend % 32 + (32 if n > 127 else 0)) | ((int(backend.toEnum()) >= 128) << 7)
                    backend_config.enable_spike_out = True
                    backend_config.enable_neuron_master = True
                    backend_config.enable_neuron_slave = False
                    backend_config.refractory_time = 5
                    backend_config.reset_holdoff = 15
                    n += 1
                else:
                    backend_config.enable_spike_out = False
                    backend_config.enable_neuron_master = False
                    backend_config.enable_neuron_slave = True
                    backend_config.refractory_time = 5
                    backend_config.reset_holdoff = 15
                backend_config.post_overwrite = False
                backend_config.connect_fire_bottom = True

                config_builder.write(
                        halco.NeuronBackendConfigOnDLS(backend, backend_block),
                        backend_config,
                        haldls.Backend.Omnibus
                        )

    def configure_neurons(self, config_builder, record_neuron=None, record_target="membrane"):
        """
        Configure neurons.
        """
        self.logger.info("configure_neurons")

        # setup neurons
        default_neuron_config = haldls.NeuronConfig()
        for neuron in halco.iter_all(halco.NeuronConfigOnDLS):
            config_builder.write(neuron, default_neuron_config, haldls.Backend.Omnibus)

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
                neuron_config.enable_reset_multiplication = False
                neuron_config.enable_leak_multiplication = False
                neuron_config.membrane_capacitor_size = 0
                neuron_config.enable_threshold_comparator = False
                neuron_config.enable_fire = False

            if int(neuron.toEnum()) == record_neuron:
                neuron_config.enable_readout = True
                neuron_config.enable_readout_amplifier = True
                neuron_config.readout_source = getattr(neuron_config.ReadoutSource, record_target)
            else:
                neuron_config.enable_readout = False

            config_builder.write(neuron, neuron_config, haldls.Backend.Omnibus)

    def configure_neuron_cells(self, config_builder):
        self.logger.info("configure_neuron_cells")

        # set capmem cells
        neuron_cells = {
                halco.CapMemRowOnCapMemBlock.v_leak: self.v_leak,
                halco.CapMemRowOnCapMemBlock.v_syn_exc: self.v_syn_exc,
                halco.CapMemRowOnCapMemBlock.v_syn_inh: self.v_syn_inh,
                halco.CapMemRowOnCapMemBlock.v_leak_adapt: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.v_threshold: self.v_thres,
                halco.CapMemRowOnCapMemBlock.v_reset: self.v_reset,
                halco.CapMemRowOnCapMemBlock.i_bias_leak: self.tau_m,
                halco.CapMemRowOnCapMemBlock.i_bias_reset: hp.jitter(1010),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_res: self.tau_syn_exc,
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: hp.jitter(1010),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_res: self.tau_syn_inh,
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: hp.jitter(1010),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_sd: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_res: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_source_follower: hp.jitter(500),
                halco.CapMemRowOnCapMemBlock.i_offset: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_readout: np.full(halco.NeuronConfigOnDLS.size, 1000),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_adapt_amp: np.zeros(halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_nmda: np.zeros(halco.NeuronConfigOnDLS.size)
                }

        neuron_cells[halco.CapMemRowOnCapMemBlock.i_bias_leak][256:] = 0
        neuron_cells[halco.CapMemRowOnCapMemBlock.i_bias_reset][256:] = 0

        for row, values in neuron_cells.items():
            for block in halco.iter_all(halco.CapMemBlockOnDLS):
                for col in halco.iter_all(halco.CapMemColumnOnCapMemBlock):
                    if int(col) > 127:
                        continue
                    config_builder.write(
                            halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock(col, row), block),
                            haldls.CapMemCell(int(values[block.value() * 128 + col.value()])),
                            haldls.Backend.Omnibus
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
                        haldls.Backend.Omnibus
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
            config_builder.write(k, haldls.CapMemCell(v), haldls.Backend.Omnibus)

    def configure(self, pre_hook=None, post_hook=None):
        config_builder = stadls.PlaybackProgramBuilder()

        if pre_hook is not None:
            pre_hook(config_builder)

        self.configure_capmem(config_builder)
        self.configure_neuron_backends(config_builder)
        self.configure_neurons(config_builder)
        self.configure_neuron_cells(config_builder)
        self.configure_readout(config_builder)
        self.configure_synapses(config_builder)
        self.configure_routing(config_builder)
        self.configure_synapse_drivers(config_builder)

        if post_hook is not None:
            post_hook(config_builder)

        config_builder.write(halco.TimerOnDLS(), haldls.Timer())
        config_builder.block_until(halco.TimerOnDLS(), 10000*fisch.fpga_clock_cycles_per_us)

        stadls.run(self.connection, config_builder.done())

    def load_and_start_ppu_program(self, program_path,
                                   mu,
                                   sparsity_offset, update_offset,
                                   random_seed, sparsity_seed):
        self.logger.info("load_and_start_ppu_program")
        
        # load PPU program
        elf_file = lola.PPUElfFile(program_path)
        elf_symbols = elf_file.read_symbols()
        self._ppu_signal_coordinate = elf_symbols["command"].coordinate
        self._ppu_mu_coordinate = elf_symbols["mu"].coordinate
        self._ppu_sparsity_offset_coordinate = elf_symbols["sparsity_offset"].coordinate
        self._ppu_update_offset_coordinate = elf_symbols["update_offset"].coordinate
        self._ppu_random_seed_coordinate = elf_symbols["random_seed"].coordinate
        self._ppu_sparsity_seed_coordinate = elf_symbols["sparsity_seed"].coordinate
        self._ppu_ppu_coordinate = elf_symbols["ppu"].coordinate
 
        program = elf_file.read_program()
        program_on_ppu = halco.PPUMemoryBlockOnPPU(
            halco.PPUMemoryWordOnPPU(0),
            halco.PPUMemoryWordOnPPU(program.size() - 1)
        )

        builder = stadls.PlaybackProgramBuilder()

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
        mu = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(mu << 24))
        update_offset = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(update_offset << 24))
        sparsity_offset = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(sparsity_offset << 24))
        random_seed = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(random_seed << 24))
        sparsity_seed = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(sparsity_seed << 24))
        for ppu in halco.iter_all(halco.PPUOnDLS):
            ppu_id = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(int(ppu.toEnum()) << 24))
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_ppu_coordinate[0], ppu), ppu_id)
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_mu_coordinate[0], ppu), mu)
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_sparsity_offset_coordinate[0], ppu), sparsity_offset)
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_update_offset_coordinate[0], ppu), update_offset)
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_random_seed_coordinate[0], ppu), random_seed)
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_sparsity_seed_coordinate[0], ppu), sparsity_seed)

        # ensure PPU is in run state
        ppu_control_reg_run = haldls.PPUControlRegister()
        ppu_control_reg_run.inhibit_reset = True

        for reg in halco.iter_all(halco.PPUControlRegisterOnDLS):
            builder.write(reg, ppu_control_reg_run)
        
        stadls.run(self.connection, builder.done())

    def stop_ppu_program(self):
        self.logger.info("stop_ppu_program")
        
        ppu_control_reg_end = haldls.PPUControlRegister()
        ppu_control_reg_end.inhibit_reset = False

        builder = stadls.PlaybackProgramBuilder()
        builder.write(halco.PPUControlRegisterOnDLS(), ppu_control_reg_end)

        stadls.run(self.connection, builder.done())

    def set_readout(self, record_neuron, readout_target="membrane"):
        """
        Configure readout chain to route analog output of given neuron to MADC.
        """
        self.logger.info("set_readout")

        builder = stadls.PlaybackProgramBuilder()

        hemisphere = record_neuron // 256
        is_odd = (record_neuron % 2) == 1
        is_even = (record_neuron % 2) == 0

        config = haldls.ReadoutSourceSelection()
        sm = config.SourceMultiplexer()
        sm.neuron_odd[halco.HemisphereOnDLS(hemisphere)] = is_odd
        sm.neuron_even[halco.HemisphereOnDLS(hemisphere)] = is_even
        config.set_buffer(halco.SourceMultiplexerOnReadoutSourceSelection(1), sm)
        config.enable_buffer_to_pad[halco.SourceMultiplexerOnReadoutSourceSelection(1)] = True
        builder.write(halco.ReadoutSourceSelectionOnDLS(), config, haldls.Backend.Omnibus)

        self.configure_neurons(builder, record_neuron, readout_target)

        stadls.run(self.connection, builder.done())

    def set_weights(self, weights):
        builder = stadls.PlaybackProgramBuilder()

        rows = np.arange(halco.SynapseDriverOnDLS.size)
        rows_on_bus = 2*(rows // 8) + (rows % 2)

        addresses = np.tile(rows_on_bus, (halco.SynapseDriverOnDLS.size, 1)).T
        
        weights_top = weights[:halco.SynapseDriverOnDLS.size, :]
        weights_bottom = weights[halco.SynapseDriverOnDLS.size:, :]
        
        synapse_matrix = lola.SynapseMatrix()
        synapse_matrix.labels.from_numpy(addresses)
        synapse_matrix.weights.from_numpy(weights_top)
        builder.write(halco.SynramOnDLS.top, synapse_matrix, haldls.Backend.Omnibus)
        
        synapse_matrix = lola.SynapseMatrix()
        synapse_matrix.labels.from_numpy(addresses)
        synapse_matrix.weights.from_numpy(weights_bottom)
        builder.write(halco.SynramOnDLS.bottom, synapse_matrix, haldls.Backend.Omnibus)

        stadls.run(self.connection, builder.done())

    def get_weights(self):
        builder = stadls.PlaybackProgramBuilder()

        tickets = list()
        for synram in halco.iter_all(halco.SynramOnDLS):
            tickets.append(builder.read(synram))

        stadls.run(self.connection, builder.done())
        
        weights = np.zeros((halco.SynapseRowOnDLS.size,
                            halco.NeuronConfigOnDLS.size // 2), dtype=np.int)
        for i, ticket in enumerate(tickets):
            weights[(halco.SynapseRowOnDLS.size // 2) * i: (halco.SynapseRowOnDLS.size // 2) * (i + 1), :] = ticket.get().weights.to_numpy()
        return weights

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
            builder.block_until(halco.TimerOnDLS(), 5000 * fisch.fpga_clock_cycles_per_us)

        # set up recording -- or not
        if madc_record is True:
            # start MADC
            madc_control = haldls.MADCControl()
            madc_control.wake_up = True
            madc_control.start_recording = True
            madc_control.enable_continuous_sampling = False
            madc_control.enable_power_down_after_sampling = True
            madc_control.enable_pre_amplifier = True
            builder.write(halco.MADCControlOnDLS(), madc_control, haldls.Backend.Omnibus)

            # stop recording after the configured number of samples
            madc_control.wake_up = False
            builder.write(halco.MADCControlOnDLS(), madc_control, haldls.Backend.Omnibus)

        # sync time
        builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
        builder.write(halco.TimerOnDLS(), haldls.Timer())
        builder.block_until(halco.TimerOnDLS(), 100)

        if spike_record is True:
            event_config = haldls.EventRecordingConfig()
            event_config.enable_event_recording = True
            builder.write(halco.EventRecordingConfigOnFPGA(), event_config)
        
        if pre_hook is not None:
            pre_hook(builder)

        if input_spikes is not None:
            times = input_spikes[:, 0]
            labels = input_spikes[:, 1].astype(np.int)

            busses_to_fire = (labels // 64) % 4
            event_addresses = labels % 64 + 64 * (labels // 128)
            event_addresses |= (1 << 7)
            event_addresses |= (1 << 13)
            event_addresses |= (busses_to_fire << 11)

            gonzales.generate_spiketrain(builder,
                                         times,
                                         event_addresses,
                                         busses_to_fire)

        builder.block_until(
                halco.TimerOnDLS(),
                int((10 * duration) * 1e6 * fisch.fpga_clock_cycles_per_us))
        
        if post_hook is not None:
            post_hook(builder)

        if spike_record is True:
            event_config = haldls.EventRecordingConfig()
            event_config.enable_event_recording = False
            builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        program = builder.done()
        stadls.run(self.connection, program)

        # get spikes
        if spike_record is True:
            self.logger.info("Found {} spikes".format(len(program.spikes)))
            spikes = program.spikes.to_numpy()
            network_spikes, input_spikes = hp.make_spikearray(spikes)
        else:
            input_spikes = None
            network_spikes = None

        if madc_record is True:
            madc_samples = program.madc_samples.to_numpy()
            samples = np.zeros((madc_samples.size, 2), dtype=np.float)
            samples[:, 0] = madc_samples["value"]
            samples[:, 1] = madc_samples["chip_time"]

            samples[:,1] /= (float(fisch.fpga_clock_cycles_per_us) / 1e6)
            samples = samples[10:, :]
        else:
            samples = None

        return input_spikes, network_spikes, samples

    def set_inhibitory_mask(self, inh_mask):
        self.inhibitory_mask = inh_mask

        config_builder = stadls.PlaybackProgramBuilder()
        self.configure_synapse_drivers(config_builder)
        stadls.run(self.connection, config_builder.done())
