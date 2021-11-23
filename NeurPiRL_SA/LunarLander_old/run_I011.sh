#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=16000M      # increase as needed
#SBATCH --cpus-per-task=16
#SBATCH --time=2:10:00
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main.py -e "Imitation" -time 7200 -seed ${seed} -oracle ${oracle} --aug_dsl -approach "1"
