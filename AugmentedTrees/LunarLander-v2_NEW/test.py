import gym
import numpy as np

env = gym.make("LunarLander-v2")

t1 = [0.26, -0.1,  -0.07, -0.38, -0.76,  1.01,  0.54, -0.1,  -0.14]
t2 = [0.2, -0.17,  0.13, -0.28,  0.38,  0.55,  0.23, -0.02,  0.43]


def my_program(ob):
    if np.cdot(t1, [1, obs]) <= 0.36:
        if np.cdot(t2, [1, obs]) <= 0.17:
            return 1
        else:
            return 0
    else:
        if np.cdot(t2, [1, obs]) <= 0.22:
            return 2
        return 3


ob = env.reset()


averaged = 0.0
for i in range(100):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        action = my_program(ob, best_program, relu_programs)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    averaged += reward
averaged /= 100.


