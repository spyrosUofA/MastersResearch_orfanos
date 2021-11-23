import sys
import numpy as np
import gym
import torch
import torch.nn as nn
import pickle
from DQN.dqn_agent import Agent
from DSL import ReLU

import os
print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
np.random.seed(seed)

# Task setup
env = gym.make('LunarLander-v2')
env.seed(seed)
o_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
action_space = np.arange(env.action_space.n)

# load DQN network
agent = Agent(state_size=8, action_size=4, seed=None)
agent.qnetwork_local.load_state_dict(torch.load('./DQN/checkpoint_complete.pth'))
neurons = 64 # len(relu_neuron[0].bias)

# Extract weigths and biases, then save each as a ReLU program
programs = []
accepted_nodes = []

for i in range(neurons):
    w = agent.qnetwork_local.fc1.weight[i].detach().numpy().tolist()
    b = agent.qnetwork_local.fc1.bias[i].detach().numpy().tolist()

    accepted_nodes.append([w, b])

#pickle.dump(programs, file=open("ReLU_programs.pickle", "wb"))
#pickle.dump(accepted_nodes, file=open("ReLU_accepted_nodes.pickle", "wb"))
print((accepted_nodes))

print(len(accepted_nodes))

# VALIDATION..
# Outputs from first hidden layer
relu_neuron = nn.Sequential(
    nn.Linear(o_dim, neurons),
    nn.ReLU()
)

# Copy weights of to first hidden layer
with torch.no_grad():
    relu_neuron[0].weight = agent.qnetwork_local.fc1.weight
    relu_neuron[0].bias = agent.qnetwork_local.fc1.bias

