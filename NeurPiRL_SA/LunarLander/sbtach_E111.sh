#!/bin/bash

for iter in {1..15}; do

 if [ $(( $iter % 4 )) -eq 1 ]; then
   oracle="ONE"
 elif [ $(( $iter % 4 )) -eq 2 ]; then
   oracle="TWO"
 elif [ $(( $iter % 4 )) -eq 3 ]; then
   oracle="THREE"
 else
   oracle="FOUR"
 fi

 ip="D110/sa_cpus-16_n-25_c-None_run-${iter}.pkl"

 sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle},ip=${ip} run_E111.sh

done

