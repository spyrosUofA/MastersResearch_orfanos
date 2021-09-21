import pickle
import gym
import os
import numpy as np
import random
from collections import deque



states = np.load("Observations.npy").tolist()
print(len(states))


q = deque(states, maxlen=5)
print(q)

print()

q.append(states[-1])
print(q)