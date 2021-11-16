#!/bin/bash

for i in {3..15}; do
 sbatch --export=scheme="${scheme}",i=${i} run_DQN.sh
done
