#!/bin/bash

for oracle in {4..15}; do
	for iter in {1..30}; do 
		sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle} run_D000.sh
	done
done
