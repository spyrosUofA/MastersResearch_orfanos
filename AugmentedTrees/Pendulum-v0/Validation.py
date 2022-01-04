import gym
from stable_baselines3 import SAC, DDPG, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.policies import obs_as_tensor

import matplotlib.pyplot as plt

import torch
import numpy as np
import os
import pickle

def predict_proba(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# create environment
seed = 1
env = gym.make("Pendulum-v0")
env.seed(seed)

# completion requirements
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=230, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

# train oracle
# Actor: 1 hidden layer size 8
model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[1], vf=[128, 128])]))


print(model.policy.state_dict())

obs = env.reset()

obs_tensor, _ = model.policy.obs_to_tensor(obs)

actions, values, logprob = model.policy.forward(obs_tensor)


print(actions)
values, log_prob, entropy = model.policy.evaluate_actions(obs_tensor, actions)



# actions must be a pytorch tensor
#log_prob = distribution.log_prob(actions)


#model.learn(int(1e6), callback=eval_callback)
value_fn = True
if value_fn:
    # From Input Layer to 1st hidden layer (8 inputs to 1 neuron)
    w1 = model.policy.state_dict()['mlp_extractor.value_net.0.weight'].detach().numpy()
    b1 = model.policy.state_dict()['mlp_extractor.value_net.0.bias'].detach().numpy()

    # max(0, W1x +b1)
    neurons = []
    for i, _ in enumerate(b1):
        neuron = np.maximum(0.0, np.dot(w1[i], obs) + b1[i])
        neurons.append(neuron)  # layer_size x 1

    # Now we are at the first hidden layer.
    w2 = model.policy.state_dict()['mlp_extractor.value_net.2.weight'].detach().numpy()
    b2 = model.policy.state_dict()['mlp_extractor.value_net.2.bias'].detach().numpy()

    neurons_h2 = [] # ?
    for i, _ in enumerate(b2):
        neuron_h2 = np.maximum(0, np.dot(w2[i], neurons) + b2[i])
        neurons_h2.append(neuron_h2)

    # output layer
    w3 = model.policy.state_dict()['value_net.weight'].detach().numpy()
    b3 = model.policy.state_dict()['value_net.bias'].detach().numpy()

    value = 0

    for i, _ in enumerate(b3):
        value += np.dot(neurons_h2, w3[i]) + b3[i]

    print("My value", value)
    values, log_prob, entropy = model.policy.evaluate_actions(obs_tensor, actions)
    print("Official Value", values)
    print()


policy_fn = True
if policy_fn:
    # From Input Layer to 1st hidden layer (8 inputs to 1 neuron)
    w1 = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1 = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()

    # max(0, W1x +b1)
    neurons = []
    for i, _ in enumerate(b1):
        neuron = np.maximum(0.0, np.dot(w1[i], obs) + b1[i])
        neurons.append(neuron)  # layer_size x 1

    # Now we are at the first hidden layer. One more layer to go!
    w2 = model.policy.state_dict()['action_net.weight'].detach().numpy()
    b2 = model.policy.state_dict()['action_net.bias'].detach().numpy()

    outputs = [] # ?
    for i, _ in enumerate(b2):
        output = np.dot(w2[i], neurons) + b2[i]
        outputs.append(output)

    # We used up all the weights. What did we get? I expect log_probs of each actions? Do I apply softmax?
    print(outputs)

    actions, values, logprob = model.policy.forward(obs_tensor, deterministic=True)
    print(actions)



    # How to get output of model??
    #print(predict_proba(model, obs)) # How to access action prob distribution :(
