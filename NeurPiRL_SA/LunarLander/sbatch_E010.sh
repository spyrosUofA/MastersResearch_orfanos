#!/bin/bash

eval_fn="Environment"
relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_ONE.pkl"

for iter in {1..15}; do

 if [ $(( $iter % 4 )) -eq 1 ]; then
   relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_ONE.pkl"
 elif [ $(( $iter % 4 )) -eq 2 ]; then
   relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_TWO.pkl"
 elif [ $(( $iter % 4 )) -eq 3 ]; then
   relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_THREE.pkl"
 else
   relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_FOUR.pkl"
 fi

 sbatch --export=scheme="${scheme}",eval_fn=${eval_fn},seed=${iter},relu=${relu} run_E010.sh

done
