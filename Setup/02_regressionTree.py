import numpy as np
from sklearn import tree

import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.RandomState(1)


# Load trajectories
trajs = pd.read_csv("trajectory.csv")

X = trajs[['x', 'v', 'theta', 'omega']]
y1 = trajs['N1']
y2 = trajs['N2']

# Neuron 1
regr_1 = tree.DecisionTreeRegressor(max_depth=2, random_state=1)
regr_1.fit(X, y1)


print(tree.plot_tree(regr_1))

tree_rules = tree.export_text(regr_1, feature_names=list(X.columns))
print(tree_rules)


# Neuron 2
regr_2 = tree.DecisionTreeRegressor(max_depth=2, random_state=1)
regr_2.fit(X, y2)

print(tree.plot_tree(regr_2))

tree_rules = tree.export_text(regr_2, feature_names=list(X.columns))
print(tree_rules)

