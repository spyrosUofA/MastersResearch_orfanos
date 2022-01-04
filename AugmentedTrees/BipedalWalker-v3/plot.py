import numpy as np
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

    depths_X = [2, 3, 4]
    mean_Y = np.mean(rewards_Y, axis=0)
    std_Y = np.std(rewards_Y, axis=0) #* (nb_seeds ** -0.5)
    print(mean_Y, std_Y)
    return depths_X, mean_Y, std_Y


def plot_all(configs, seeds):

    for count, config in enumerate(configs):
        depths_X, mean_Y, std_Y = data_oracle(config, seeds)
        plt.plot(depths_X, mean_Y)
        plt.fill_between(depths_X, mean_Y - std_Y, mean_Y + std_Y, alpha=0.2)

    plt.xlabel('Tree Depth')
    plt.ylabel('Reward (100 episodes)')
    plt.title('LunarLander-v2')
    plt.ylim([-200, 250])
    plt.hlines(200., 2, 3, colors='red', linestyles='dashed')
    plt.legend(configs, loc='lower right')
    plt.pause(20)
    plt.savefig('TreePlot.png', dpi=1080, bbox_inches="tight")


plot_all(["4x0", "32x0", "64x64", "256x256", "BaseDSL"], 15)


