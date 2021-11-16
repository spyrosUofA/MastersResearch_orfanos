#!/bin/bash

for iter in {16..20}; do
	for oracle in {1..15}; do 
		sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle} run_E010.sh
	done
done
