#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=16000M      # increase as needed
#SBATCH --cpus-per-task=16
#SBATCH --time=0:10:00
module load python/3.6

python 1_plot.py -config "D010" -nb_oracles 15 -nb_resets 30
python 1_plot.py -config "D000" -nb_oracles 15 -nb_resets 30
python 1_plot.py -config "D100" -nb_oracles 10 -nb_resets 30
python 1_plot.py -config "D110" -nb_oracles 2 -nb_resets 30
python 1_plot.py -config "E010_D010" -nb_oracles 15 -nb_resets 30
