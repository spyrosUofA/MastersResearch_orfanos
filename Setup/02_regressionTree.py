import numpy as np
from sklearn import tree
import pandas as pd
import re

# Load trajectories
trajs = pd.read_csv("trajectory.csv")

X = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']]
y1 = trajs['N1']
y2 = trajs['N2']

# Regression tree for Neuron 1
regr_1 = tree.DecisionTreeRegressor(max_depth=3, random_state=1)
regr_1.fit(X, y1)
tree_rules = tree.export_text(regr_1, feature_names=list(X.columns))
print(tree_rules)

# Extract decision rules as strings
decision_rules = tree_rules.replace("|--- value:", "return")
decision_rules = decision_rules.replace("|---", "if")
decision_rules = decision_rules.replace("|", "")

print(decision_rules)
boolean_rules = re.findall(r'if (.*)', decision_rules)

"""
print(np.mean(trajs.query('o[1] <= -0.17 & o[0] <= -0.37 & o[3] <= 0.70')['a']))
print(np.mean(trajs.query('o[1] <= -0.17 & o[0] <= -0.37 & o[3] > 0.70')['a']))

print(np.mean(trajs.query('o[1] <= -0.17 & o[0] > -0.37 & o[0] <= 0.27')['a']))
print(np.mean(trajs.query('o[1] <= -0.17 & o[0] > -0.37 & o[0] <= 0.27')['a']))

print(np.mean(trajs.query('o[1] > -0.17 & o[0] <= -0.33 & o[1] <= 0.03')['a']))
print(np.mean(trajs.query('o[1] > -0.17 & o[0] <= -0.33 & o[1] > 0.03')['a']))

print(np.mean(trajs.query('o[1] > -0.17 & o[0] > -0.33 & o[1] <= -0.04')['a']))
print(np.mean(trajs.query('o[1] > -0.17 & o[0] > -0.33 & o[1] > -0.04')['a']))
"""

# Regression tree for Neuron 2
regr_2 = tree.DecisionTreeRegressor(max_depth=2, random_state=1)
regr_2.fit(X, y2)
tree_rules = tree.export_text(regr_2, feature_names=list(X.columns))
print(tree_rules)

# Extract decision rules
decision_rules = tree_rules.replace("|--- value:", "return")
decision_rules = decision_rules.replace("|---", "if")
decision_rules = decision_rules.replace("|", "")
boolean_rules.extend(re.findall(r'if (.*)', decision_rules))

# Only keep unique boolean expressions
boolean_rules = list(set(boolean_rules))

print("Decision Rules:")
print(boolean_rules)
exit()
np.savetxt("boolean_rules.txt", boolean_rules, fmt='%s')
