#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=16000M      # increase as needed
#SBATCH --cpus-per-task=16
#SBATCH --time=0:10:00
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install matplotlib

python plot.py -config "D010" -nb_oracles 5 -nb_resets 30
python plot.py -config "D000" -nb_oracles 5 -nb_resets 30
