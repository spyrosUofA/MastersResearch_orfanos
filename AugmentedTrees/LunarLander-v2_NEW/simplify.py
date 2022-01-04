import copy
import pickle


def ite_same_same(p, changes):
    if isinstance(p, Ite):
        if isinstance(p.children[1], AssignAction) and isinstance(p.children[2], AssignAction):
            if p.children[1].children[0] == p.children[2].children[0]:
                p = p.children[2]
                changes += 1
    return p, changes


def num_lt_num(p, changes):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], Num) and isinstance(child.children[1], Num):
                if child.children[0].children[0] < child.children[1].children[0]:
                    p = p.children[1]
                    changes += 1
                    return p, changes
                else:
                    p = p.children[2]
                    changes += 1
                    return p, changes
    return p, changes


def num_plus_num(p, changes):
    if isinstance(p, Addition):
        if isinstance(p.children[0], Num) and isinstance(p.children[1], Num):
            p = Num.new(p.children[0].children[0] + p.children[1].children[0])
            changes += 1
    return p, changes


def num_times_num(p, changes):
    if isinstance(p, Multiplication):
        if isinstance(p.children[0], Num) and isinstance(p.children[1], Num):
            p = Num.new(p.children[0].children[0] * p.children[1].children[0])
            changes += 1
            return p, changes
    return p, changes


def plus_zero(p, changes):
    if isinstance(p, Addition):
        if isinstance(p.children[0], Num):
            if p.children[0].children[0] == 0:
                p = p.children[1]
                changes += 1
                return p, changes
        elif isinstance(p.children[1], Num):
            if p.children[1].children[0] == 0:
                p = p.children[0]
                changes += 1
                return p, changes
    return p, changes


def times_zero(p, changes):
    if isinstance(p, Multiplication):
        if isinstance(p.children[0], Num):
            if p.children[0].children[0] == 0:
                p = Num.new(0.0)
                changes += 1
                return p, changes
        elif isinstance(p.children[1], Num):
            if p.children[1].children[0] == 0:
                p = Num.new(0.0)
                changes += 1
                return p, changes
    return p, changes


def relu_lt_0(p, changes):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], ReLU) and isinstance(child.children[1], Num):
                if child.children[1].children[0] <= 0:
                    p = p.children[2] # false case
                    changes += 1
                    return p, changes
    return p, changes


def negative_lt_relu(p, changes):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], Num) and isinstance(child.children[1], ReLU):
                if child.children[0].children[0] < 0:
                    p = p.children[1] # true case
                    changes += 1
                    return p, changes
    return p, changes


def simplify_node(p, changes):
    c0 = changes
    program = copy.deepcopy(p)
    program, changes = ite_same_same(program, changes)
    program, changes = times_zero(program, changes)
    program, changes = plus_zero(program, changes)
    program, changes = num_plus_num(program, changes)
    program, changes = num_times_num(program, changes)
    program, changes = num_lt_num(program, changes)
    program, changes = relu_lt_0(program, changes)
    program, changes = negative_lt_relu(program, changes)
    return program, changes - c0


def simplify_program(p, changes):

    for i in range(p.get_number_children()):
        simplified, change = simplify_node(p.children[i], changes)
        changes += change
        p.replace_child(simplified, i)
        # simplify children
        if isinstance(p.children[i], Node):
            changes = simplify_program(p.children[i], changes)
            #changes += simplify_program(p.children[i], changes)
            #changes += simplify_program(p.children[i], 0)
    return changes


def simplify_program1(p):
    while True:
        changes = simplify_program(p, 0)
        #print(p.to_string() + '\n')
        if changes == 0:
            return p


from DSL import *
from evaluation import *
import time
import numpy as np

#policy = pickle.load(open("../binary_programs/D000/Oracle-15/sa_cpus-16_n-25_c-None_run-8.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-13/sa_cpus-16_n-25_c-None_run-3.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-12/sa_cpus-16_n-25_c-None_run-3.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-4/sa_cpus-16_n-25_c-None_run-25.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-9/sa_cpus-16_n-25_c-None_run-9.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-8/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))




# My program



def my_program1(obs):


    n0 = [[-0.29006934, -0.3880698, -0.6872576, -0.948045, 2.3667905, 1.0735272, -0.7814564, 0.40444276], 0.39982024]
    n1 = [[-0.06402702, 1.4401745, 0.26052123, 2.0634193, -0.07485023, -0.5543935, 1.1538936, 1.1730388], 0.5495856]
    n2 = [[-0.243671, 1.2416056, -1.3031385, 1.3099798, 0.92196876, 1.3039432, 0.07569284, 0.5164872], 0.5016362]
    n3 = [[-1.4197779, 0.14036536, -1.2065054, -1.3177942, 1.9234407, 1.6862239, 0.12679574, 0.01929791], 0.30844527]
    n4 = [[0.45367214, 0.01314157, -0.20283164, -1.7964867, -1.1939923, -1.7043539, -0.1739508, -0.73991656], 0.8354575]
    n5 = [[1.3430973, 1.0496248, 1.3290908, -0.9635685, -2.2084415, -1.2899541, 0.22435127, 0.5361045], 0.55276954]
    n6 = [[-0.40309227, 0.30767497, -0.6145687, -0.15801342, 2.2621489, 1.5403486, 0.5670489, -0.724267], 0.32253602]
    n7 = [[0.0608627, -1.0698364, -0.24970861, 0.6567928, 0.086555, -0.22911389, 1.8958944, 1.5869105], 0.2918102]

    n0 = max(0, np.dot(obs, n0[0]) + n0[1])
    n1 = np.dot(obs, n1[0]) + n1[1]
    n2 = np.dot(obs, n2[0]) + n2[1]
    n3 = np.dot(obs, n3[0]) + n3[1]
    n4 = max(0, np.dot(obs, n4[0]) + n4[1])
    n5 = np.dot(obs, n5[0]) + n5[1]
    n6 = np.dot(obs, n6[0]) + n6[1]
    n7 = np.dot(obs, n7[0]) + n7[1]

    if (n1 > 1) or (n7 > 3.): # and (np.dot(obs, n5[0]) + n5[1]) > 0:
        return 0
    elif n4 * 0.79554075 + 0.552772 > 1.: # and n5 > 1:
        return 2
    elif (n0 * -1.3802716 + -0.15940964 <= -0.3): #or n5 > 2: # n1 > 0.5 and n4 > 0.8 and n5 > 1. : #  and (np.dot(obs, n4[0]) + n4[1]) > .6:
        return 1

    elif n4 * 0.79554075 + 0.552772 < -1 and n5 > 0.34:
        return 3
    else:
        print("?")
        return 0


import numpy as np
from sklearn import tree
import pandas as pd
import re
X = np.load("./Oracle/4x0/1/neurons.npy") #.tolist()
Y = np.load("./Oracle/4x0/1/neurons_to_actions.npy") #.tolist()

# Regression tree for Neuron 1
regr_1 = tree.DecisionTreeClassifier(max_depth=150, random_state=1)
regr_1.fit(X, Y)
print(regr_1)

def my_program(obs):
    n0 = [[ 0.28382367,  1.7934723,  -0.11373619,  1.584884,    0.23984677, -0.42233634, 1.0650071,   2.0341644 ], 0.5301271]
    n1 = [[ 0.36524048, -0.19711488, -0.16623792, -2.815942,   -0.7019816,  -0.32718712, -0.26316372, -0.2660255 ], 0.42466605]
    n2 = [[-1.2129432,   0.37068072, -0.8900809,  -0.8597776,   2.3637958,   2.5205786, -0.40225548, -0.35761788], 0.4344356]
    n3 = [[ 0.38897833,  0.45173603,  1.38248,    -0.85729873, -2.597608 ,  -0.97952, -0.29168594,  0.31902564], 0.40432847]

    n0 = max(0, np.dot(obs, n0[0]) + n0[1])
    n1 = max(0, np.dot(obs, n1[0]) + n1[1])
    n2 = max(0, np.dot(obs, n2[0]) + n2[1])
    n3 = max(0, np.dot(obs, n3[0]) + n3[1])

    a = regr_1.predict([[n0, n1, n2, n3]])[0]
    print(a)
    return a
    if n1 <= 0.36:
        if n0 <= 3.37:
            return 0
        else:
            return 0




    if obs[6] + obs[7] > 0:
        return 0

    if n0 * 1.21 + 0.31 < -2:
        return 2

    if n2 * -1.63728 + -0.22375107 < -1:
        return 1

    if n0 * 1.21 + 0.31 > 1:
        return 0


    if n3 * -1.2812959 + -0.16854282 < -1.5:
        return 3

    print("HELLO")
    return 2

import gym
env = gym.make("LunarLander-v2")



steps = 0
averaged = 0.0
render = True
games = 5

for i in range(games):
    ob =env.reset()
    reward = 0.0
    while True:
        if render:
            env.render()
        action = my_program(ob)
        print(action)
        ob, r_t, done, _ = env.step(action)
        steps += 1
        reward += r_t

        if done:
            env.close()
            break
    averaged += reward

print(averaged / games)
exit()




#policy = pickle.load(open("OLD/binary_programs/D110_D110/64x64/2/sa_cpus-1_n-100_c-5000_run-21.pkl", "rb"))

policy = pickle.load(open("./binary_programs/D110/32x32/1/sa_cpus-1_n-100_c-None_run-1.pkl", "rb"))






#policy = pickle.load(open("./binary_programs/D110/64x64/1/sa_cpus-1_n-100_c-5000_run-122.pkl", "rb"))
print("MEAN:", Environment({}, 200, 1, "Pendulum-v0").collect_reward(policy, 100, True))

exit()


policy = pickle.load(open("OLD/binary_programs/D110/256x0/2/sa_cpus-1_n-100_c-5000_run-21.pkl", "rb"))






#print(policy.to_string() + '\n')
p_new = simplify_program1(policy)
#print(p_new.to_string() + '\n')

print("MEAN:", Environment({}, 200, 1, "Pendulum-v0").collect_reward(p_new, 16, True))



ite = policy.children[0]
print(ite.to_string())
ite2 = ite.children[2]
print(ite2.to_string())
false_case = ite2.children[1]
print(false_case.to_string())
##redundant_true = false_case.children[0]
#print(redundant_true.to_string())
#false_case.replace_child(AssignAction.new(-0.6791540746132676), 0)

ite2.replace_child(AssignAction.new(-0.6791540746132676), 1)
#exit()

print(p_new.to_string() + '\n')
#print("------------------")
print("MEAN:", Environment({}, 200, 1, "Pendulum-v0").collect_reward(p_new, 200, True))
print("MEAN REWARD (100 episodes):", Environment({}, 100, 1, "Pendulum-v0").evaluate(p_new))

print(p_new.to_string() + '\n')

exit()


exit()


rew = 0
for i in range(1, 16, 1):
    policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-" + str(i) + "/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))

    print(policy.to_string() + '\n')
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    print(policy.to_string() + '\n')

    rew += Environment({}, 200, 1).evaluate(policy)

print(rew / 15.0)
# E010_D010_old: 163.48063343676603

