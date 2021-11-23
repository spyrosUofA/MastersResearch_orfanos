import copy
import pickle
from evaluation import Environment
import torch
import numpy as np
from ActorCritic.model import ActorCritic
from DSL import *
import gym

# Load neural Policy
ORACLE_NB = "THREE"

oracle = ActorCritic()
oracle.load_state_dict(torch.load("../LunarLander/ActorCritic/Oracle/" + ORACLE_NB + "/Policy.pth"))
# Load Trajectory
inputs = np.load("../LunarLander/ActorCritic/Oracle/" + ORACLE_NB + "/Observations.npy").tolist()
actions = np.load("../LunarLander/ActorCritic/Oracle/" + ORACLE_NB + "/Actions.npy").tolist()
# Load ReLUs
accepted_relus = pickle.load(open("../LunarLander/ActorCritic/Oracle/" + ORACLE_NB + "/ReLUs.pkl", "rb"))

# Arguments for evaluation function
parameters_oracle = {"oracle": oracle,
                     "inputs": inputs,
                     "actions": actions,
                     "ReLUs": accepted_relus,
                     "capacity": None}

# Load PiRL Policy
#policy = pickle.load(open("./good_SA_program.pkl", "rb"))
policy = pickle.load(open("../LunarLander/binary_programs/E010_D010_old/sa_cpus-16_n-25_c-None_run-5.pkl", "rb"))
#policy = StartSymbol().new(Ite().new(Lt.new(Num.new(0.668), Num.new(0.666)), AssignAction.new(0), AssignAction.new(1)))
#policy = pickle.load(open("../LunarLander/binary_programs/Eval-DAgger_BayesOpt-False_ReLU-True_InitProg-False/sa_cpus-1_n-25_c-50000_run-0Queue.pkl", "rb"))
print(policy.to_string())


eval_fn = Environment(copy.deepcopy(parameters_oracle), 10, 0)
#print(eval_fn.collect_reward(policy, 50))
print(eval_fn.find_distance(policy))

correct = 0.0
for i in range(len(inputs)):
    if policy.interpret({'obs': inputs[i], 'act': 0}) == parameters_oracle['actions'][i]: #oracle(np.array(inputs[i])):
        correct += 1
print(correct / len(inputs))
exit()


#print(policy.interpret({'obs': inputs[0], 'act': 0}))
#print(oracle(np.array(inputs[0])))

def get_action(obs, p):
    actions = []
    for ob in obs:
        namespace = {'obs': ob, 'act': 0}
        p.interpret(namespace)
        actions.append(namespace['act'])
    return actions


irl_action = get_action(inputs, policy)

correct = 0.0
for i in range(len(inputs)):
    if policy.interpret({'obs': inputs[i], 'act': 0}) == irl_action[i]:
        correct += 1
print(correct / len(inputs))




