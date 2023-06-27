#!/bin/bash
echo "Poweroff"
srun -p cube --wafer 68 --fpga-without 3 --gres cubex8 -c 1 --mem 1G -t 5:00 --pty -- singularity exec --app visionary-dls /containers/stable/latest hxcube_enable_fpga.py 3 --poweroff

echo "Poweron"
srun -p cube --wafer 68 --fpga-without 3 --gres cubex8 -c 1 --mem 1G -t 5:00 --pty -- singularity exec --app visionary-dls /containers/stable/latest hxcube_enable_fpga.py 3 --skip-bootloader

echo "Done"
