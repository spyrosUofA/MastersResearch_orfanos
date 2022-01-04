import numpy as np

import gym.envs.box2d.lunar_lander
from gym.envs.box2d.lunar_lander import INITIAL_RANDOM
globals()['INITIAL_RANDOM'] = 1500

setattr(gym.envs.box2d.lunar_lander, 'INITIAL_RANDOM', 1000)




env = gym.make("LunarLander-v2")
env.seed(0)


t1 = [0.26, -0.1,  -0.07, -0.38, -0.76,  1.01,  0.54, -0.1,  -0.14]
t2 = [0.2, -0.17,  0.13, -0.28,  0.38,  0.55,  0.23, -0.02,  0.43]

t1 = [0.26, 0, 0, -0.38, -0.76,  1.01,  0.54, 0, 0]
t2 = [0.2, -0.17, 0.13, -0.28,  0.38,  0.55,  0.23, 0.5, 0.5]

def my_program(obs):
    obs = obs.tolist()
    obs.insert(0, 1)
    if np.dot(t1, obs) <= 0.36:
        if np.dot(t2, obs) <= 0.17:
            return 1
        return 0
    else:
        if np.dot(t2, obs) <= 0.22:
            return 2
        return 3


ob = env.reset()


averaged = 0.0
games = 10
for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = my_program(ob)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    averaged += reward
    #env.close()
averaged /= games

print(averaged)






t1 = [0, 0, -0.38, -0.76,  1.01,  0.54, 0, 0]
t2 = [-0.17, 0.13, -0.28,  0.38,  0.55,  0.23, 0.5, 0.5]

def my_program(obs):
    obs = obs.tolist()
    #obs.insert(0, 1)
    if np.dot(t1, obs) <= 0.1:
        if np.dot(t2, obs) <= -0.03:
            print(1)
            return 1
        return 0
    else:
        if np.dot(t2, obs) <= 0.02:
            return 2
        return 3

env = gym.make("LunarLander-v2")
env.seed(0)
ob = env.reset()


averaged = 0.0
games = 10
for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = my_program(ob)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    averaged += reward
    #env.close()
averaged /= games

print(averaged)


# ORIGINAL
env = gym.make("LunarLander-v2")
env.seed(0)
ob = env.reset()


t1 = [0.26, -0.1,  -0.07, -0.38, -0.76,  1.01,  0.54, -0.1,  -0.14]
t2 = [0.2, -0.17,  0.13, -0.28,  0.38,  0.55,  0.23, -0.02,  0.43]

#t1 = [0.26, 0, 0, -0.38, -0.76,  1.01,  0.54, 0, 0]
#t2 = [0.2, -0.17, 0.13, -0.28,  0.38,  0.55,  0.23, 0.5, 0.5]

def my_program(obs):
    obs = obs.tolist()
    obs.insert(0, 1)
    if max(0, np.dot(t1, obs)) <= 0.36:
        if max(0, np.dot(t2, obs)) <= 0.17:
            return 1
        return 0
    else:
        if max(0, np.dot(t2, obs)) <= 0.22:
            return 2
        return 3

averaged = 0.0
games = 10
for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = my_program(ob)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    averaged += reward
    #env.close()
averaged /= games

print(averaged)
