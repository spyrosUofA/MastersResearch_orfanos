#!/bin/bash

for oracle in {1..1}; do
	for iter in {1..1}; do
		sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle} run_D010.sh
	done
done
