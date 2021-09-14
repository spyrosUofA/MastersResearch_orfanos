from model import ActorCritic
import torch
import gym
from PIL import Image
#from NeurPiRL_SA.LunarLander.DSL import ReLU
import pickle
import numpy as np
import os

def test(n_episodes=50, model='TWO'):

    # Make environment
    env = gym.make('LunarLander-v2')

    # Load Pretrained policy
    policy = ActorCritic()
    name = 'LunarLander_' + model + '.pth'
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))

    if not os.path.exists('./Oracle/' + model):
        os.makedirs('./Oracle/' + model)

    # Save ReLU programs
    programs = []
    for i in range(128):
        w = policy.affine.weight[i].detach().numpy()
        b = policy.affine.bias[i].detach().numpy()
        programs.append([w, b])
    pickle.dump(programs, file=open('./Oracle/' + model + '/ReLUs.pkl', "wb"))

    # Save policy as ActorCritic() object
    torch.save(policy.state_dict(), './Oracle/' + model + '/Policy.pth')

    render = False
    save_gif = True

    obs = []
    actions = []

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            ###
            obs.append(state)
            actions.append(action)
            ###
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()

    # Save Trajectories
    np.save(file='./Oracle/' + model + '/Observations.npy', arr=obs)
    np.save(file='./Oracle/' + model + '/Actions.npy', arr=actions)


if __name__ == '__main__':
    test(50, "ONE")
    test(50, "TWO")
    test(50, "THREE")
    test(50, "FOUR")
