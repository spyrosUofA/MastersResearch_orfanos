import numpy as np
from sklearn import tree
import pandas as pd
import re
import gym
import matplotlib.pyplot as plt
import torch

def collect_reward(env, num_eps, policy):

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
        avg_rew = collect_reward(num_eps, tree_policy)
        avg_rew_policies.append(avg_rew)

    return avg_rew_policies

def generate_plots1(num_eps, demos, tree_depth):

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

def dagger_rollout(env, policy, oracle, roll_outs=1):

    obs = []
    actions_oracle = []
    rew_eps = []

    for _ in range(roll_outs):
        ob = env.reset()
        print(ob)
        reward = 0
        while True:
            obs.append(ob)

            # oracle's action (expectation)
            actions_oracle.append(np.random.choice(a=[0, 1], p=oracle(torch.FloatTensor(ob)).detach().numpy()))

            # student's action (reality)
            action = policy.predict([ob])[0]
            #namespace = {'obs': ob, 'act': 0}
            #p.interpret(namespace)
            #action = [namespace['act']]
            ob, r_t, done, _ = env.step(action)

            reward += r_t

            if done:
                rew_eps.append(reward)
                print("DAgger Reward: " + str(reward))
                break

    return np.array(obs).tolist(), actions_oracle, np.mean(rew_eps)

def algo_NDPS(pomd, oracle, initial_hist, tree_size, nb_updates, roll_outs=1, seed=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Task setup
    env = gym.make(pomd)
    env.seed(seed)
    obs_space = np.arange(env.observation_space.shape[0])
    action_space = np.arange(env.action_space.n)

    # load Oracle policy
    pi_oracle = torch.load(oracle)
    print(pi_oracle)

    # initial trajectory from oracle
    trajs = pd.read_csv("../Setup/trajectory.csv", nrows=initial_hist)
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy().tolist()
    actions = trajs['a'].to_numpy().tolist()

    # initial irl policy
    tree_policy = tree.DecisionTreeClassifier(max_depth=tree_size, random_state=seed)
    tree_policy.fit(observations, actions)

    # evaluate initial 
    rew = collect_reward(env, 25, tree_policy)

    rewards = [rew]

    for i in range(nb_updates):
        print(i)

        # roll out IRL policy, collect imitation data
        obs, act_oracle, rews = dagger_rollout(env, tree_policy, pi_oracle, roll_outs)

        # DAgger style imitation learning (update histories)
        observations.extend(obs)
        actions.extend(act_oracle)

        # derive IRL policy with updated history
        tree_policy = tree.DecisionTreeClassifier(max_depth=tree_size, random_state=seed)
        tree_policy.fit(observations, actions)

        # evaluate policy
        avg = collect_reward(env, 25, tree_policy)
        rewards.append(avg)
        print("Reward: " + str(avg) + "\n")

    """
    plt.clf()
    plt.plot(range(len(rewards)), rewards)
    plt.title("CartPole-v1: Imitating Neural Policy with Regression Tree of Depth " + str(tree_size))
    plt.ylim(0, 500)
    plt.ylabel("Average Return")
    plt.xlabel("DAgger Roll Outs")
    #plt.legend(title="# Demonstrations", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    """
    return rewards

def generate_plots(pomd, oracle, initial_hist, tree_size, nb_updates, roll_outs=1, seed=1):

    i = 0
    rows = 2
    cols = int(np.ceil(len(tree_size) / rows))

    fig, axs = plt.subplots(rows, cols)
    cmap = plt.get_cmap("tab10")

    for r in range(rows):
        for c in range(cols):
            reward = algo_NDPS(pomd, oracle, initial_hist, tree_size[i], nb_updates)
            axs[r, c].plot(range(len(reward)), reward, color=cmap(i))
            axs[r, c].set_title("Tree Depth: " + str(tree_size[i]))
            axs[r, c].set_ylim([0, 500])
            axs[r, c].hlines(475, 0, nb_updates, linestyles="dashed", color='pink')
            axs[r, c].xaxis.set_ticks([j for j in range(0, nb_updates+10, 10)])

            i += 1

            if i == len(tree_size):
                break

    fig.suptitle("CartPole-v1: NDPS with Regression Trees")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.set(xlabel='DAgger Rollouts', ylabel='Average Return')
        ax.label_outer()

    plt.show()


oracle = '../Setup/ppo_2x4_policy.pth'
pomd = "CartPole-v1"
initial_hist = 500
#tree_size = [2**x for x in range(1, 7)]
tree_size = [8]
nb_updates = 100

#algo_NDPS(pomd, oracle, 500, 16, 100)

generate_plots(pomd, oracle, initial_hist, tree_size, nb_updates)