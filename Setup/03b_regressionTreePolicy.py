import numpy as np
from sklearn import tree
import pandas as pd
import re
import gym

# Load trajectories
trajs = pd.read_csv("trajectory.csv")

X = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']]
y1 = trajs['a']

# Regression tree for Neuron 1
regr_1 = tree.DecisionTreeClassifier(max_depth=2048, random_state=1)
regr_1.fit(X, y1)
tree_rules = tree.export_text(regr_1, feature_names=list(X.columns))
print(tree_rules)

# Task setup
env = gym.make('CartPole-v1')
env.seed(1)
obs = []

# Generate a trajectory
o = env.reset()
done = False
rew_this = 0
rew_ep = []

for i in range(10000):

    # state
    obs.append(o)

    # take action
    a = regr_1.predict([o])[0]

    # Observe, transition
    op, r, done, infos = env.step(a)

    # log
    rew_this += r

    # Update environment
    if done:
        # log
        rew_ep.append(rew_this)

        # reset
        o = env.reset()
        rew_this = 0

    else:
        o = op

print(rew_ep)
print(np.mean(rew_ep))