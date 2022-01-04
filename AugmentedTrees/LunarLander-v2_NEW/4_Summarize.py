import numpy as np
import argparse
import matplotlib.pyplot as plt
from shutil import copyfile

def data_oracle(oracle, nb_seeds):

    rewards_Y = []
    for i in range(nb_seeds):
        name = str(oracle) + '/' + str(i + 1 + 15)
        if oracle == "BaseDSL":
            rewards_Y.append(np.load("./Oracle/" + name + "/BaseTreeRewards.npy").tolist())
        else:
            rewards_Y.append(np.load("./Oracle/" + name + "/AugTreeRewards.npy").tolist()[0:2])
            print(np.load("./Oracle/" + name + "/AugTreeRewards.npy").tolist())

    depths_X = [2] #, 3, 4]
    mean_Y = np.mean(rewards_Y, axis=0)
    std_Y = np.std(rewards_Y, axis=0) #* (nb_seeds ** -0.5)
    print(oracle, mean_Y, std_Y)
    return depths_X, mean_Y, std_Y


def print_table(configs, seeds):
    for count, config in enumerate(configs):
        depths_X, mean_Y, std_Y = data_oracle(config, seeds)


print_table(["4x0"], 14)


