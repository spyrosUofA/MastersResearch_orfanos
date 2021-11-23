import pickle
from evaluation import *
import pandas as pd
import numpy as np


if False:
    trajs = pd.read_csv("../LunarLander/trajectory.csv")
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]', 'o[4]', 'o[5]', 'o[6]', 'o[7]']].to_numpy()
    actions = trajs['a'].to_numpy()
    scorer = Imitation(25)
    scorer.add_trajectory(observations, actions)

    p1 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_InitialProgram.pkl", "rb"))
    print("Score 1: ",  scorer.evaluate(p1)[0])

    p2 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))
    print("Score 2: ",  scorer.evaluate(p2)[0])

    p3 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test5.pkl", "rb"))
    print("Score 3: ", scorer.evaluate(p3)[0])
    print()
    #exit()


if False:
    p1 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_InitialProgram.pkl", "rb"))
    print("Avg. Reward 1: ",  Environment(25).evaluate(p1)[0])

    p2 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))
    print("Avg. Reward 2: ", Environment(25).evaluate(p2)[0])

    p3 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test5.pkl", "rb"))
    print("Avg. Reward 3: ", Environment(25).evaluate(p3)[0])
    print()
    exit()



#print(policy.to_string())
#avg_reward = Environment("", "", 1, 0).eval_render(policy)[0]
#print("\nAverage reward", avg_reward)



import torch
import numpy as np
from ActorCritic.model import ActorCritic


# Load neural Policy
oracle = ActorCritic()
oracle.load_state_dict(torch.load("../LunarLander/ActorCritic/Oracle/ONE/Policy.pth"))
# Load Trajectory
inputs = np.load("../LunarLander/ActorCritic/Oracle/ONE/Observations.npy").tolist()
actions = np.load("../LunarLander/ActorCritic/Oracle/ONE/Actions.npy").tolist()
# Load ReLUs
accepted_relus = pickle.load(open("../LunarLander/ActorCritic/Oracle/ONE/ReLUs.pkl", "rb"))
# Arguments for evaluation function
parameters_oracle = {"oracle": oracle,
                     "inputs": inputs,
                     "actions": actions,
                     "ReLUs": accepted_relus,
                     "capacity": 100}

#score, games = Imitation(parameters_oracle, 1, 0).evaluate(policy)
#print(score, games)
#score = Imitation(parameters_oracle, 1, 0).collect_reward(policy)
#print(score)
nb_games = 10
eval_fn = Environment(parameters_oracle, nb_games, 0)

from DSL import *

#policy = StartSymol().new(Ite().new(Lt.new(Num.new(0.668), Num.new(0.666)), AssignAction.new(0), AssignAction.new(1)))
#policy = StartSymbol().new(Ite().new(Lt.new(Multiplication.new(Num.new(2.0), Observation.new(3)), Observation.new(4)), AssignAction.new(2), AssignAction.new(1)))
#policy = pickle.load(open("o5r11.pkl", "rb"))
#policy = pickle.load(open("nice_policy_d010.pkl", "rb"))
policy = pickle.load(open("HELLO.pkl", "rb"))

print(policy.to_string())
print(eval_fn.collect_reward(policy, nb_games, False))
print(eval_fn.evaluate(policy))
