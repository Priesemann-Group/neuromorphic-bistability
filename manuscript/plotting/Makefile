.PHONY: all data

FIG_DIR=../build/figures
DATA_DIR=../data
NAMES=rate_evolution \
      final_rate \
      weight_histogram \
      connectivity \
      average_weight \
      ac \
      activity_distribution \
      activity \
      timescale_comparison \
      colorbar \
      simulation_ac \
      simulation_activity_distribution \
      theory_potential \
      theory_activity \
      v_leak \
      v_thres \
      v_reset \
      tau_syn_exc \
      tau_syn_inh \
      tau_m \
      epsp_height \
      ipsp_height \
      parametrization \
      phase_full_emul_60 \
      phase_full_emul_80 \
      phase_full_emul_100 \
      phase_full_emul_colorbar \
      phase_full_emul_cross \
      phase_full_sim_60 \
      phase_full_sim_80 \
      phase_full_sim_100 \
      phase_full_sim_colorbar \
      phase_full_sim_cross \
      phase_rate \
      phase_taus \
      phase_fano \
      phase_cv \
      phase_activity_distribution_0 \
      phase_activity_distribution_1 \
      phase_activity_distribution_2 \
      phase_activity_distribution_3 \
      phase_raster_0 \
      phase_raster_1 \
      phase_raster_2 \
      phase_raster_3 \
      avalanche_distribution \
      si_emul_final_rate \
      si_sim_final_rate \
      si_sim_offset_final_rate \
      si_emul_weight_histogram \
      si_sim_weight_histogram \
      si_sim_offset_weight_histogram \
      si_emul_connectivity \
      si_sim_connectivity \
      si_sim_offset_connectivity \
      si_emul_average_weight \
      si_sim_average_weight \
      si_sim_offset_average_weight \
      si_emul_ac \
      si_sim_ac \
      si_sim_offset_ac \
      si_emul_activity_distribution \
      si_sim_activity_distribution \
      si_sim_offset_activity_distribution \
      si_emul_activity \
      si_sim_activity \
      si_sim_offset_activity \
      si_emul_timescale_comparison \
      si_sim_timescale_comparison \
      si_sim_offset_timescale_comparison \
      si_hom_final_rate \
      si_hom_connectivity \
      si_hom_weight_histogram \
      si_hom_average_weight \
      si_hom_ac \
      si_hom_activity_distribution \
      si_hom_activity

PGF_FIGURES=$(NAMES:%=$(FIG_DIR)/%.pgf)
PDF_FIGURES=$(NAMES:%=$(FIG_DIR)/%.pdf)
DATA=$(NAMES:%=$(DATA_DIR)/%.npz)

all: directories $(PGF_FIGURES) $(PDF_FIGURES)

directories:
	mkdir -p $(FIG_DIR)

clean:
	rm $(PGF_FIGURES)
	rm $(PDF_FIGURES)

$(FIG_DIR)/%.pgf : plot_%.py $(DATA_DIR)/%.npz matplotlibrc
	python3 -u $< $(word 2,$^) --save $@

$(FIG_DIR)/%.pdf : plot_%.py $(DATA_DIR)/%.npz matplotlibrc
	python3 -u $< $(word 2,$^) --save $@
