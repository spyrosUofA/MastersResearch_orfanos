#!/bin/bash

eval_fn="Environment"
relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_ONE.pkl"

for iter in {1..5}; do

  if [ $(( $iter % 4 )) -eq 1 ]; then
    relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_ONE.pkl"
    echo "remainder 1"
  elif [ $(( $iter % 4 )) -eq 2 ]; then
    relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_TWO.pkl"
    echo " is remainder 2"
  elif [ $(( $iter % 4 )) -eq 3 ]; then
    relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_THREE.pkl"
    echo " is remainder 2"
  else
    relu="../LunarLander/ActorCritic/ReLU_programs/ReLUs_FOUR.pkl"
    echo " is remainder 0"
  fi

  sbatch eval_fn=${eval_fn},seed=${iter},relu=${relu} run_boF_env_reluT_initF.sh

done
