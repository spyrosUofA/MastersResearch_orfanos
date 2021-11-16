#!/bin/bash

for oracle in {10..10}; do
	for iter in {30..30}; do 
		sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle} run_D100.sh
	done
done
