import pickle
from evaluation import Evaluate, Environment, Imitation, DAgger
import time
import torch
import numpy as np
from ActorCritic.model import ActorCritic
from DSL import *
import copy

# SA policy
policy = pickle.load(open("../LunarLander/binary_programs/Eval-DAgger_BayesOpt-False_ReLU-True_InitProg-False/sa_cpus-1_n-25_c-50000_run-0Queue.pkl", "rb"))
#policy = StartSymbol().new(Ite().new(Lt.new(Num.new(0.668), Num.new(0.666)), AssignAction.new(0), AssignAction.new(1)))

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

eval_fn = Environment(copy.deepcopy(parameters_oracle), 100, 0)
t0 = time.time()
score = eval_fn.eval_render(policy, False)
print(score, time.time() - t0)

eval_fn = Environment(copy.deepcopy(parameters_oracle), 100, 0)
t0 = time.time()
score = eval_fn.collect_reward(policy)
print(score, time.time() - t0)


eval_fn = Imitation(copy.deepcopy(parameters_oracle), 100, 0)
t0 = time.time()
score = eval_fn.collect_reward(policy)
print(score, time.time() - t0)
