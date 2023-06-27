#include <array>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fisch/vx/constants.h"
#include "haldls/vx/v1/neuron.h"
#include "haldls/vx/v1/timer.h"
#include "stadls/vx/v1/init_generator.h"
#include "stadls/vx/v1/playback_program.h"
#include "stadls/vx/v1/playback_program_builder.h"

#include "lola/vx/synapse.h"
#include "lola/vx/cadc.h"

using namespace halco::common;
using namespace halco::hicann_dls::vx;
using namespace haldls::vx::v1;
using namespace stadls::vx::v1;
using namespace lola::vx;

namespace py = pybind11;

void generate_spiketrain(
	PlaybackProgramBuilder& builder,
	py::array_t<double> times,
	py::array_t<uint16_t> neuron_labels,
	py::array_t<uint16_t> spl1_addresses
	) {

	double previous_time = 0.0;

	for(size_t i=0; i<times.size(); ++i) {
		if(times.at(i) > previous_time) {
			builder.block_until(
				TimerOnDLS(),
				Timer::Value(times.at(i) * 1e6 * fisch::vx::fpga_clock_cycles_per_us));

			previous_time = times.at(i);
		}
		
		std::array<SpikeLabel, 1> labels;
		labels.at(0).set_neuron_label(NeuronLabel(neuron_labels.at(i)));
		labels.at(0).set_spl1_address(SPL1Address(spl1_addresses.at(i)));

		SpikePack1ToChip pack;
		pack.set_labels(labels);
		builder.write(SpikePack1ToChipOnDLS(), pack);
	}
}

void reset_correlation(PlaybackProgramBuilder& builder) {
	for(auto row : iter_all<CorrelationResetRowOnDLS>())
		builder.write(row, CorrelationResetRow());
}

typedef std::vector<PlaybackProgram::ContainerTicket<CADCSampleRow>> cadc_tickets_type;
cadc_tickets_type measure_correlation(PlaybackProgramBuilder& builder) {
	cadc_tickets_type tickets;
	for(auto row : iter_all<CADCSampleRowOnDLS>())
		tickets.push_back(builder.read(row));
	return tickets;
}

PYBIND11_MODULE(gonzales, m) {
	py::module::import("pystadls_vx_v1");
	m.def("generate_spiketrain", &generate_spiketrain, "Generate a playback program builder for inserting spikes.");
	m.def("reset_correlation", &reset_correlation, "Reset all correlation rows.");
	m.def("measure_correlation", &measure_correlation, "Measure correlation for all synapses.");
}
