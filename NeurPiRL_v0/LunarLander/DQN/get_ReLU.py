import sys
import numpy as np
import gym
import torch
import torch.nn as nn
import pandas as pd
from dqn_agent import Agent



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
np.random.seed(seed)

# Task setup
env = gym.make('LunarLander')
env.seed(seed)
o_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
action_space = np.arange(env.action_space.n)


# load DQN network
agent = Agent(state_size=8, action_size=4, seed=None)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_complete.pth'))
neurons = 64


# Outputs from first hidden layer
relu_neuron = nn.Sequential(
    nn.Linear(o_dim, neurons),
    nn.ReLU()
)

print(agent.qnetwork_local.state_dict())

# Copy weights of to first hidden layer
with torch.no_grad():
    relu_neuron[0].weight = agent.qnetwork_local.fc1.weight
    relu_neuron[0].bias = agent.qnetwork_local.fc1.bias

# weights and biases
for i in range(neurons):
    w = relu_neuron[0].weight[i].detach().numpy()
    b = relu_neuron[0].bias[i].detach().numpy()

    print(b)
