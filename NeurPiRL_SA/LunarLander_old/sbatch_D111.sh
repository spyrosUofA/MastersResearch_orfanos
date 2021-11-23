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

 sbatch --export=scheme="${scheme}",seed=${iter},oracle=${oracle} run_D111.sh

done

