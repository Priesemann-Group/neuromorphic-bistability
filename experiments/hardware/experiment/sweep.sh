#!/bin/bash

case "$1" in
    config)
        # Create config file
        srun -p cube --wafer 68 --fpga-without 3 --mem 8G -t 5:00 --pty -- \
        singularity exec --app visionary-dls /containers/stable/latest \
	    python -u test.py \
            --program_path ../ppu/build/homeostasis.bin \
            --use_calibration \
            --prefix ../data/calib
        ;;
    weights)
        # Set up sweep and run weights
        srun -p cube --wafer 68 --fpga-without 3 --mem 8G -t 5:00:00 --pty -- \
        singularity exec --app visionary-dls /containers/stable/latest \
	    python -u sweep.py \
            ../data/calib_config.json \
            ../data/weights/sweep_calib \
            --key0 freq \
            --key1 seed \
            --ax0log \
            --min0 6e2 \
            --min1 728804 \
            --max0 20e3 \
            --max1 728903 \
            --step0 5 \
            --step1 1
        ;;
    static)
        # Static experiments
        srun -p cube --wafer 68 --fpga-without 3 --mem 8G -t 5:00:00 --pty -- \
        singularity exec --app visionary-dls /containers/stable/latest \
	    python -u sweep_static.py \
            ../data/weights/sweep_calib_config.json \
            ../data/static/sweep_calib_static 
        ;;
    letter)
        srun -p cube --wafer 68 --fpga-without 3 --mem 8G -t 5:00:00 --pty -- \
        singularity exec --app visionary-dls /containers/stable/latest \
	    python -u sweep_letter.py \
            ../data/calib_config.json \
            ../data/letter/sweep_calib \
            --key0 freq \
            --key1 seed \
            --ax0log \
            --min0 6e2 \
            --min1 728804 \
            --max0 20e3 \
            --max1 728903 \
            --step0 5 \
            --step1 1 \
            --n_overlay 5 \
	    --sigma_t 5e-6
        ;;
    audio)
        # srun -p cube --wafer 68 --fpga-without 3 --mem 8G -t 5:00:00 --pty -- \
        # singularity exec --app visionary-dls /containers/stable/latest \
	    python -u sweep_audio.py \
            ../data/calib_config.json \
            ../data/audio/sweep_calib \
            --key0 freq \
            --key1 seed \
            --ax0log \
            --min0 4e3 \
            --min1 728804 \
            --max0 20e3 \
            --max1 728903 \
            --step0 3 \
            --step1 1 \
            --stimuli_path ../plots/data/shd
        ;;
    pert)
        srun -p cube --wafer 68 --fpga-without 3 --mem 8G -t 10:00:00 --pty -- \
        singularity exec --app visionary-dls /containers/stable/latest \
        python -u sweep_pert.py \
            ../data/weights/sweep_calib_config.json \
            ../data/pert/sweep_calib_pert
        ;;
    traces)
        srun -p cube --wafer 68 --fpga-without 3 --mem 8G -t 5:00:00 --pty -- \
        singularity exec --app visionary-dls /containers/stable/latest \
        python -u sweep_traces.py \
            ../data/weights/sweep_calib_config.json \
            ../data/traces/sweep_calib_traces \
            --record_neuron 0 \
            --record_target inh_synin
        ;;
    *)
        echo "Usage: $0 {config|weights|static|letter|pert|traces}"
        exit 1
esac
