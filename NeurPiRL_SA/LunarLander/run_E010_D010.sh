#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=8000M      # increase as needed
#SBATCH --cpus-per-task=1
#SBATCH --time=4:30:00
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

echo ${ip}
#python main.py -time 7800 -seed ${seed} -oracle ${oracle} -e "DAgger" --aug_dsl -approach "0"
python main.py -time 7800 -seed ${seed} -oracle ${oracle} -e "Environment" --aug_dsl -approach "0" -ip ${ip}

