#!/bin/bash

for oracle in {10..10}; do
	for iter in {30..30}; do 
		ip="D010/Oracle-$oracle/sa_cpus-16_n-25_c-None_run-$iter.pkl"
		sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle},ip=${ip} run_E010_D010.sh
	done
done
