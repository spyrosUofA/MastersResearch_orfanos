#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=8000M      # increase as needed
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
module load python/3.6

python 1_plot.py -config "D110/2x4" -nb_resets 2
python 1_plot.py -config "D110/64x64" -nb_resets 2
python 1_plot.py -config "D100/64x64" -nb_resets 2
#python 1_plot.py -config "D000"
#python 1_plot.py -config "D100" -nb_oracles 10
#python 1_plot.py -config "D110" -nb_oracles 2
#python 1_plot.py -config "E010_D010_old"

python 2_compare.py
