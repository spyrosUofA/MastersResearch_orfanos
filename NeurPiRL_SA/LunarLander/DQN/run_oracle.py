from dqn_agent import Agent
import torch
import gym
import pickle
import numpy as np
import os

def test(model=1, nb_episodes=1):
    
    # Create directory where we will save trajectory and actor agent
    path_to = './Oracle/' + str(model)
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    # Load Pretrained policy
    agent = Agent(state_size=8, action_size=4, seed=model)
    agent.qnetwork_local.load_state_dict(torch.load(path_to + '/Policy.pth'))

    # Save ReLU programs
    programs = []
    nb_neurons = len(agent.qnetwork_local.fc1.weight)

    for i in range(nb_neurons):
        w = agent.qnetwork_local.fc1.weight[i].detach().numpy()
        b = agent.qnetwork_local.fc1.bias[i].detach().numpy()

        programs.append([w, b])
    pickle.dump(programs, file=open(path_to + '/ReLUs.pkl', "wb"))

    # Save policy as Agent() object
    #torch.save(agent.qnetwork_local.state_dict(), path_to + '/Policy.pth')

    # Generate and save experience
    env = gym.make('LunarLander-v2')
    env.seed(model)

    obs = []
    actions = []

    for i_episode in range(1, nb_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action, _ = agent.act(state)
            # Record Trajectory
            obs.append(state)
            actions.append(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            running_reward += reward

            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()

    # Save Trajectories
    np.save(file=path_to + '/Observations.npy', arr=obs)
    np.save(file=path_to + '/Actions.npy', arr=actions)


if __name__ == '__main__':
    test()

