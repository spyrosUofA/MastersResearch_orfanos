import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
from DQN.dqn_agent import Agent
from DQN.run_oracle import test
import os
import sys


def dqn(seed=0, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.005, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    print(seed)

    env = gym.make('LunarLander-v2')
    env.seed(seed)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    agent = Agent(state_size=8, action_size=4, seed=seed)

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    steps = 0
    times = [time.time()]
    # torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/' + str(0) + '.pth')
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
            #torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/'+ str(i_episode) +'_220.pth')

        if np.mean(scores_window) >= 250.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)), '\tsteps: {:.2f}'.format(steps))

            # Save results
            save_to = './Oracle/' + str(seed)
            if not os.path.exists(save_to):
                os.makedirs(save_to)

            # Save Learning Curve #1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.savefig(save_to + '/learning_curve_1.png')
            plt.clf()

            # Save Learning Curve #2
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #plt.plot(times, scores)
            #plt.ylabel('Score')
            #plt.xlabel('Episode #')
            #plt.savefig(save_to + '/learning_curve_2.png')

            # Save Network
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            torch.save(agent.qnetwork_local.state_dict(), save_to + '/Policy.pth')

            # Save trajectories
            test(seed, 1)
            env.close()
            break

    return scores, times

if __name__ == '__main__':
   dqn()

   #for i in range(15):
   #     dqn(i)
