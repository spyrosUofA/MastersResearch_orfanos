import numpy as np
from sklearn import tree
import pandas as pd
import re
import gym
import matplotlib.pyplot as plt

def evaluate_policy(num_eps, policy):
    env = gym.make('CartPole-v1')
    env.seed(1)
    rew_eps = []

    for ep in range(num_eps):
        # reset at start of each episode
        o = env.reset()
        done = False
        rew_this = 0

        while not done:
            # state
            #obs.append(o)

            # take action
            a = policy.predict([o])[0]

            # Observe, transition
            op, r, done, infos = env.step(a)
            o = op

            # log
            rew_this += r

        # end of episode
        rew_eps.append(rew_this)

    return np.mean(rew_eps)

def get_avg_rew(num_eps, demos, tree_depth):
    trajs = pd.read_csv("trajectory.csv", nrows=demos)
    X = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']]
    y = trajs['a']
    avg_rew_policies = []

    for alpha in tree_depth:
        # train policy
        tree_policy = tree.DecisionTreeClassifier(max_depth=alpha, random_state=1)
        tree_policy.fit(X, y)

        # evaluate policy
        avg_rew = evaluate_policy(num_eps, tree_policy)
        avg_rew_policies.append(avg_rew)

    return avg_rew_policies

def generate_plots(num_eps, demos, tree_depth):

    i = 0
    rows = 2
    cols = int(np.ceil(len(demos) / rows))

    fig, axs = plt.subplots(rows, cols)
    cmap = plt.get_cmap("tab10")

    for r in range(rows):
        for c in range(cols):
            reward = get_avg_rew(num_eps, demos[i], tree_depth)
            axs[r, c].plot(tree_depth, reward, color=cmap(i))
            axs[r, c].set_title(str(demos[i]) + " Demonstrations")
            axs[r, c].set_ylim([0, 500])
            axs[r, c].set_xscale('log', base=2)
            axs[r, c].xaxis.set_ticks(tree_depth)

            i += 1

            if i == len(demos):
                break

    fig.suptitle("CartPole-v1: Imitating PPO with Regression Trees")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.set(xlabel='Tree Depth', ylabel='Average Return')
        ax.label_outer()

    plt.show()

    """
    plt.clf()
    for t in time_steps:
        rew = get_avg_rew(num_eps, t, tree_depth)
        plt.plot(range(1, len(tree_depth) + 1), rew, label=t)
    plt.title("CartPole-v1: Imitating Neural Policy with Regression Trees")
    plt.ylim(0, 500)
    plt.ylabel("Average Return")
    plt.xlabel("Tree Depth (log 2)")
    plt.legend(title="# Demonstrations", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    """

# PARAMETERS
NUM_EPS = 25
TREE_DEPTH = [2**x for x in range(1, 10)]
TIME_STEPS = [500, 1000, 2500, 5000, 10000, 50000]

generate_plots(NUM_EPS, TIME_STEPS, TREE_DEPTH)