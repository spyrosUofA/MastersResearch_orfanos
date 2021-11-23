#!/bin/bash

for iter in {10..15}; do

 if [ $(( $iter % 4 )) -eq 1 ]; then
   oracle="ONE"
 elif [ $(( $iter % 4 )) -eq 2 ]; then
   oracle="TWO"
 elif [ $(( $iter % 4 )) -eq 3 ]; then
   oracle="THREE"
 else
   oracle="FOUR"
 fi


 ip="D010/sa_cpus-16_n-25_c-None_run-${iter}.pkl"

 sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle},ip=${ip} run_E010_D010.sh

done
