#!/bin/bash

for oracle in {1..15}; do
  for seed in {2..2}; do
    o1="2x4/$oracle"
    o2="64x64/$oracle"

    sbatch --export=scheme="${scheme}",seed=${seed},o1=${o1},o2=${o2} run_D110.sh

  done
done
