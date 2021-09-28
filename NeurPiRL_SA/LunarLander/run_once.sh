#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=1600M     # increase as needed
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main.py -e "Environment" --relu "../LunarLander/ActorCritic/ReLU_programs/ReLUs_ONE.pkl" -seed 5 -time 3600
