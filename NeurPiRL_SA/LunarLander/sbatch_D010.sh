#!/bin/bash

for oracle in {12..12}; do
	for iter in {2..2}; do
		sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle} run_D010.sh
	done
done
