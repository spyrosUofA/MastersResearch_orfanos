#!/bin/bash

for oracle in {1..2}; do
	for iter in {1..30}; do 
		sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle} run_D110.sh
	done
done
