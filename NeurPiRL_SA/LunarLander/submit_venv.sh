#!/bin/bash
#SBATCH --account=def-orfanos
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --time=1:00:00
module load python/3.6\
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip freeze > requirements.txt
pip install --no-index -r requirements.txt

cd NeurPiRL_BUS/LunarLander
python3 main.py -n 10 -log_file log_test2 -program_file program_test2 -time 3600 -search SimulatedAnnealing