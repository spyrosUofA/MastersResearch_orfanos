#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --time=1:00:00
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main.py -e ${eval_fn} --relu ${relu} -seed ${seed} -time 3600