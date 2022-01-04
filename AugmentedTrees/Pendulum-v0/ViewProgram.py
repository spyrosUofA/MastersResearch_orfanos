import numpy as np
from sklearn import tree
import pandas as pd
import re
import pickle




def relu_string(relus):
    relu_names = []
    for i, relu in enumerate(relus):
        name = '(' + str(np.around(relu[0], 2)) + " *dot* obs[:] + " + str(np.round(relu[1], 2)) + ")"
        relu_names.append(name)
    return relu_names



rews = np.load("Oracle/32x0/1/AugTreeRewards.npy").tolist()
relus = pickle.load(open("Oracle/32x0/1/ReLUs.pkl", "rb"))
#relu_names = relu_string(relus)
relu_names = ["w" + str(i).zfill(1) for i in range(64)]
relu_names.extend(["x", "y", "v_x", "v_y", "theta", "v_th", "c_l", "c_r"])

trees = pickle.load(open("Oracle/32x0/1/AugTreePrograms.pkl", "rb"))
regr_1 = trees[2]



def fast_tree1(obs, tree, relus):
    all_features = [0] * len(relus)
    all_features.extend(obs)
    used_features = [x for x in tree.tree_.feature if 0 <= x < len(relus)]

    for i in used_features:
        all_features[i] = max(0, np.dot(relus[i][0], obs) + relus[i][1])

    return tree.predict([all_features])[0]

def fast_tree(obs, tree, relus):

    all_features = [0] * len(relus)
    all_features.extend(obs)
    used_features = [x for x in tree.tree_.feature if x >= 0]

    for i in used_features:
        try:
            all_features[i] = max(0, np.dot(relus[i][0], obs) + relus[i][1])
        except:
            continue

    return tree.predict([all_features])[0]


def used_units(tree, relus):
    return [x for x in tree.tree_.feature if 0 <= x < len(relus)]


def fastest_tree(obs, tree, used_units, relus):
    all_features = [0] * len(relus)
    all_features.extend(obs)

    for i in used_units:
        all_features[i] = max(0, np.dot(relus[i][0], obs) + relus[i][1])
    return tree.predict([all_features])[0]



def get_neurons(obs, relus_list):
    neurons = []
    for i, relu in enumerate(relus_list):
        neuron = max(0, np.dot(obs, relu[0]) + relu[1])
        neurons.append(neuron)
    neurons.extend(obs)
    return neurons


def my_program(obs, tree, relus):
    neurons = get_neurons(obs, relus)
    return tree.predict([neurons])[0]

import time

M = 20000

t0 = 0
t1 = 0

t0 = time.time()
for _ in range(M):
    fast_tree([0,0,0], regr_1, relus)
t1 = time.time()
print("faster??", t1-t0)


t0 = time.time()
units = used_units(regr_1, relus)
for _ in range(M):
    fastest_tree([0,0,0], regr_1, units, relus)
t1 = time.time()
print("fastestest??", t1-t0)


t0 = time.time()
for _ in range(M):
    my_program([0,0,0], regr_1, relus)
t1 = time.time()
print("default", t1-t0)




t0 = time.time()
for _ in range(M):
    fast_tree([0,0,0], regr_1, relus)
t1 = time.time()
print("fastest??", t1-t0)



t0 = time.time()
units = used_units(regr_1, relus)
for _ in range(M):
    fastest_tree([0,0,0], regr_1, units, relus)
t1 = time.time()
print("fastestest??", t1-t0)





