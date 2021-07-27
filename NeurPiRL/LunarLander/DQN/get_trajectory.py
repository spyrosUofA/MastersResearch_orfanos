import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import pandas as pd


env = gym.make('LunarLander')
#env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from dqn_agent import Agent

agent = Agent(state_size=8, action_size=4, seed=None)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_complete.pth'))
#print(agent.qnetwork_local)

n_episodes=1
max_t=1000
eps=0.01

obs = []
actions = []


for i_episode in range(1, n_episodes + 1):
    state = env.reset()
    score = 0
    for t in range(max_t):
        action, _ = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)

        obs.append(state)
        actions.append(action)

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            break

print("Score:" + str(score))

df = pd.DataFrame(obs, columns=['o[0]', 'o[1]', 'o[2]', 'o[3]', 'o[4]', 'o[5]', 'o[6]', 'o[7]'])
df['a'] = actions
#df['N1'] = predN1
#df['N2'] = predN2
df.to_csv(path_or_buf="trajectory.csv", index=False)
print(df)


print(agent.qnetwork_local)

