from model import ActorCritic
import torch
import gym
from PIL import Image
#from NeurPiRL_SA.LunarLander.DSL import ReLU
import pickle
import pandas as pd


def test(n_episodes=50, model='TWO'):
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()

    name = 'LunarLander_' + model + '.pth'
    
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))

    # Save ReLU programs
    programs = []
    for i in range(128):
        w = policy.affine.weight[i].detach().numpy()
        b = policy.affine.bias[i].detach().numpy()

        programs.append([w, b])
    pickle.dump(programs, file=open("./ReLU_programs/ReLUs_" + model + ".pkl", "wb"))

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

    df = pd.DataFrame(obs, columns=['o[0]', 'o[1]', 'o[2]', 'o[3]', 'o[4]', 'o[5]', 'o[6]', 'o[7]'])
    df['a'] = actions
    df.to_csv(path_or_buf="./Trajectories/trajectory_" + model + ".csv", index=False)
    env.close()
            
if __name__ == '__main__':
    test(50, "ONE")
    test(50, "TWO")
    test(50, "THREE")
    test(50, "FOUR")