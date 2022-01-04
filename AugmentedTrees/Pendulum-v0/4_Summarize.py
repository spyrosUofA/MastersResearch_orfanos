import numpy as np
import argparse
import matplotlib.pyplot as plt
from shutil import copyfile

# 0: ID
# 1: Reward
# 2: Score
# 3: Nb Games Played
# 4: Times


def data_oracle(oracle, nb_seeds):

    rewards_Y = []
    for i in range(nb_seeds):
        name = str(oracle) + '/' + str(i + 1)
        if oracle == "BaseDSL":
            rewards_Y.append(np.load("./Oracle/" + name + "/BaseTreeRewards.npy").tolist())
        else:
            rewards_Y.append(np.load("./Oracle/" + name + "/AugTreeRewards.npy").tolist())

    mean_Y = np.mean(rewards_Y, axis=0)
    std_Y = np.std(rewards_Y, axis=0) #* (nb_seeds ** -0.5)
    print(oracle, mean_Y, std_Y)
    return mean_Y, std_Y


def plot_all(configs, seeds):

    for count, config in enumerate(configs):
        data_oracle(config, seeds)



plot_all(["32x0", "64x64", "256x256", "BaseDSL"], 9)


