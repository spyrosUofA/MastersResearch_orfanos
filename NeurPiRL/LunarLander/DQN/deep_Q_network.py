import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

env = gym.make('LunarLander')
#env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from dqn_agent import Agent

agent = Agent(state_size=8, action_size=4, seed=None)

# # watch an untrained agent
# state = env.reset()
# for j in range(1000):
#     action = agent.act(state)
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break
#
# env.close()

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    steps = 0
    times = [time.time()]
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/' + str(0) + '.pth')
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action, _ = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            steps+=1
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), '\tsteps: {:.2f}'.format(steps), end="")
        if i_episode % 100 == 0:
            times.append(time.time())
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), '\tsteps: {:.2f}'.format(steps))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/'+ str(i_episode) +'.pth')
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)), '\tsteps: {:.2f}'.format(steps))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_complete.pth')
            break
    return scores, times

#from eval import *
scores, times = dqn()
np.save("times_models.npy", times)
#perf = evaluate()
#np.save("performances_models.npy", perf)
#plotting_time(times, perf, "neural")
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('learning_curve_2000.png')
plt.show()
env.close()

