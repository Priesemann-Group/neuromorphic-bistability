.PHONY: all data

PREFIX=/wang/users/bcramer/cluster_home/project/paranoise/experiments/hardware/data
STATIC_CONFIG=$(PREFIX)/weights/sweep_calib_config.json
PERT_CONFIG=$(PREFIX)/pert/sweep_calib_pert_config.json
LETTER_CONFIG=$(PREFIX)/letter/sweep_calib_config.json

FIG_DIR=plots
DATA_DIR=data
NAMES=rates \
      raster \
      weights \
      ac \
      letter_delay \
      vrd

FIGURES=$(NAMES:%=$(FIG_DIR)/%.pgf)
DATA=$(NAMES:%=$(DATA_DIR)/%.npz)

all: directories $(DATA) $(FIGURES)

data: directories $(DATA)

directories:
	mkdir -p $(DATA_DIR)
	mkdir -p $(FIG_DIR)

$(FIG_DIR)/%.pgf: plot_%.py $(DATA_DIR)/%.npz matplotlibrc
	echo "$< $(word 2,$^)"
	srun -p compile -N 1 -c 1 -t 10:00 singularity exec --app visionary-dls --overlay /containers/overlays/2020-05-28_buster_texlive.img /containers/stable/latest python -u $< $(word 2,$^) --save $@

$(DATA_DIR)/rates.npz: make_rates.py
	srun -p compile -N 1 -c 1 -t 10:00 singularity exec --app visionary-dls /containers/stable/latest python -u make_rates.py \
		$(STATIC_CONFIG) \
		--save $@

$(DATA_DIR)/raster.npz: make_raster.py
	srun -p compile -N 1 -c 1 -t 10:00 singularity exec --app visionary-dls /containers/stable/latest python -u make_raster.py \
		$(STATIC_CONFIG) \
		--save $@

$(DATA_DIR)/weights.npz: make_weights.py
	srun -p compile -N 1 -c 1 -t 10:00 singularity exec --app visionary-dls /containers/stable/latest python -u make_weights.py \
		$(STATIC_CONFIG) \
		--save $@

$(DATA_DIR)/ac.npz: make_ac.py
	srun -p compile -N 1 -c 10 -t 1:00:00 singularity exec --app visionary-dls /containers/stable/latest python -u make_ac.py \
		$(STATIC_CONFIG) \
		--save $@

$(DATA_DIR)/letter_delay.npz: make_letter_delay.py
	srun -p compile -N 1 -c 16 -t 4:00:00 singularity exec --app visionary-dls /containers/stable/latest python -u make_letter_delay.py \
		$(LETTER_CONFIG) \
		--save $@

$(DATA_DIR)/vrd.npz: make_vrd.py
	srun -p compile -N 1 -c 20 -t 5:00:00 singularity exec --app visionary-dls /containers/stable/latest python -u make_vrd.py \
		$(LETTER_CONFIG) \
		--save $@
