#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=8000M      # increase as needed
#SBATCH --cpus-per-task=1
#SBATCH --time=3:20:00

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main.py -time 3660 -seed ${seed} -oracle $o1 -e "DAgger" --bo --aug_dsl -approach "0"
python main.py -time 3600 -seed ${seed} -oracle $o2 -e "DAgger" --bo --aug_dsl -approach "0"
python main.py -time 3660 -seed ${seed} -oracle $o2 -e "DAgger" --bo -approach "0"

