import sys
import numpy as np
import gym
import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
np.random.seed(seed)

# Task setup
env = gym.make('CartPole-v1')
env.seed(seed)
o_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
action_space = np.arange(env.action_space.n)

# load ppo policies
ppo_relu = torch.load("PPO_ReLU_2x4_policy.pth")
ppo_sigmoid = torch.load("PPO_Sigmoid_2x4_policy.pth")
neurons = 2

# Outputs from first hidden layer
relu_neuron = nn.Sequential(
    nn.Linear(o_dim, neurons),
    nn.ReLU()
)

sigmoid_neuron = nn.Sequential(
    nn.Linear(o_dim, neurons),
    nn.Sigmoid()
)

# Copy weights of to first hidden layer
with torch.no_grad():
    relu_neuron[0].weight = ppo_relu[0].weight
    sigmoid_neuron[0].weight = ppo_sigmoid[0].weight

for activation in ["ReLU", "Sigmoid"]:

    if activation == "ReLU":
        policy = ppo_relu
        layer_1 = relu_neuron
        file_name = "ReLU_Trajectories.csv"

    elif activation == "Sigmoid":
        policy = ppo_sigmoid
        layer_1 = sigmoid_neuron
        file_name = "Sigmoid_Trajectories.csv"


    # Now with the fixed policy, we generate some episodes for training data
    obs = []
    actions = []
    predN1 = []
    predN2 = []


    # Generate a trajectory
    o = env.reset()
    episode = 0
    for i in range(50000):

        # state, N1 and N2 (the neurons values)
        obs.append(o)
        n1, n2 = layer_1(torch.FloatTensor(o)).detach().numpy()
        predN1.append(n1)
        predN2.append(n2)

        # take action
        a = np.random.choice(a=action_space, p=policy(torch.FloatTensor(o)).detach().numpy())
        actions.append(a)

        # Observe, transition
        #env.render()
        op, r, done, infos = env.step(a)

        # Update environment
        if done:
            o = env.reset()
            episode += 1
            print(episode)
            print(i+1)
            #break
        else:
            o = op

    df = pd.DataFrame(obs, columns=['o[0]', 'o[1]', 'o[2]', 'o[3]'])
    df['a'] = actions
    df['N1'] = predN1
    df['N2'] = predN2

    df.to_csv(path_or_buf=file_name, index=False)
