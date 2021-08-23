import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import pandas as pd

seed = 10

env = gym.make('LunarLander-v2')
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from dqn_agent import Agent

agent = Agent(state_size=8, action_size=4, seed=seed)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_complete_min220.pth'))
print(agent.qnetwork_local.parameters())


#"""
network = torch.load('checkpoint_complete_min220.pth')
print(network['fc1.weight'])
#print(network['fc1.bias'])
"""
agent.qnetwork_local.fc1.weight = network['fc1.weight'].detach().numpy()
agent.qnetwork_local.fc1.bias = network['fc1.bias']
agent.qnetwork_local.fc2.weight = network['fc2.weight']
agent.qnetwork_local.fc2.bias = ['fc2.bias']
agent.qnetwork_local.fc3.weight = network['fc3.weight']
agent.qnetwork_local.fc3.bias = network['fc3.bias']
"""

n_episodes=25
max_t=1000
eps=0.015

obs = []
actions = []
scores = []


for i_episode in range(1, n_episodes + 1):
    state = env.reset()
    score = 0
    for t in range(max_t):

        action, _ = agent.act(state, eps)
        obs.append(state)
        actions.append(action)

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            print("Score:" + str(score))
            scores.append(score)
            break

print(np.mean(scores))

df = pd.DataFrame(obs, columns=['o[0]', 'o[1]', 'o[2]', 'o[3]', 'o[4]', 'o[5]', 'o[6]', 'o[7]'])
df['a'] = actions
#df.to_csv(path_or_buf="trajectory_25.csv", index=False)
print(df)