import gym
import highway_env
from stable_baselines3 import DQN

model = DQN.load("highway_dqn/model")
policy_weights = model.get_parameters()['policy']

w0 = policy_weights['q_net.q_net.0.weight']
b0 = policy_weights['q_net.q_net.0.bias']

print(len(w0), len(b0))

print(w0.shape)