import pickle
import gym
import os
import numpy as np
import random
from collections import deque



states = np.load("Observations.npy").tolist()
print(len(states))


q = deque(maxlen=5)
q.append(1, 2, 3, 4, 5, 6, 7)

print(q)