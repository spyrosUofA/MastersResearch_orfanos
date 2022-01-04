import numpy as np
import pickle
import gym
from sklearn import tree

def get_neurons(obs, relus):
    neurons = []
    for i, relu in enumerate(relus):
        neuron = max(0, np.dot(obs, relu[0]) + relu[1])
        neurons.append(neuron)
    neurons.extend(obs)
    return neurons


def my_program1(obs, tree, relus):
    neurons = get_neurons(obs, relus)
    return tree.predict([neurons])[0]

def my_program(obs, trees, relus):
    action = []
    for _, tree in enumerate(trees):
        all_features = [0] * len(relus)
        all_features.extend(obs)
        used_features = [x for x in tree.tree_.feature if 0 <= x < len(relus)]
        for i in used_features:
            all_features[i] = max(0, np.dot(relus[i][0], obs) + relus[i][1])
        action.append(tree.predict([all_features])[0])
    return action


policy = pickle.load(open("./Oracle/256x0/0/AugTreePrograms.pkl", "rb"))[0]
relus = pickle.load(open("./Oracle/256x0/0/ReLUs.pkl", "rb"))

policy = pickle.load(open("./Oracle/128x128/1/AugTreePrograms3.pkl", "rb"))[0]
relus = pickle.load(open("./Oracle/128x128/1/ReLUs.pkl", "rb"))

print(tree.export_text(policy[0]))

env = gym.make("BipedalWalker-v3")
env.seed(0)
ob = env.reset()

averaged = 0.0
games = 100

actions = list()

for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        env.render()
        action = my_program(ob, policy, relus)
        actions.append(action[0])
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    print(reward)
    actions = list(set(actions))
    averaged += reward
    env.close()
averaged /= games

print(games, averaged)