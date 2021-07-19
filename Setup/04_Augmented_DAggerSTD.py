import numpy as np
from sklearn import tree
import pandas as pd
import time
import gym
import matplotlib.pyplot as plt
import torch

def collect_reward(env, policy, num_eps):

    rew_eps = []

    for ep in range(num_eps):
        # reset at start of each episode
        o = env.reset()
        done = False
        rew_this = 0

        while not done:
            # state
            X = pd.DataFrame([o])
            X.columns = ['o[0]', 'o[1]', 'o[2]', 'o[3]']
            # Addition
            X['o[0]+o[1]'] = X['o[0]'] + X['o[1]']
            X['o[0]+o[2]'] = X['o[0]'] + X['o[2]']
            X['o[0]+o[3]'] = X['o[0]'] + X['o[3]']
            X['o[1]+o[2]'] = X['o[1]'] + X['o[2]']
            X['o[1]+o[3]'] = X['o[1]'] + X['o[3]']
            X['o[2]+o[3]'] = X['o[2]'] + X['o[3]']
            # Multiplication
            X['o[0]*o[0]'] = X['o[0]'] * X['o[0]']
            X['o[1]*o[1]'] = X['o[1]'] * X['o[1]']
            X['o[2]*o[2]'] = X['o[2]'] * X['o[2]']
            X['o[3]*o[3]'] = X['o[3]'] * X['o[3]']
            X['o[0]*o[1]'] = X['o[0]'] * X['o[1]']
            X['o[0]*o[2]'] = X['o[0]'] * X['o[2]']
            X['o[0]*o[3]'] = X['o[0]'] * X['o[3]']
            X['o[1]*o[2]'] = X['o[1]'] * X['o[2]']
            X['o[1]*o[3]'] = X['o[1]'] * X['o[3]']
            X['o[2]*o[3]'] = X['o[2]'] * X['o[3]']

            # take action
            a = policy.predict(X)[0]
            #a = policy.predict([o])[0]

            # Observe, transition
            op, r, done, infos = env.step(a)
            o = op

            # log
            rew_this += r

        # end of episode
        rew_eps.append(rew_this)

    return np.mean(rew_eps), np.std(rew_eps)


def dagger_rollout(env, policy, oracle, roll_outs=1):

    obs = []
    actions_oracle = []
    rew_eps = []

    for _ in range(roll_outs):
        ob = env.reset()
        reward = 0
        while True:

            # oracle's action (expectation)
            actions_oracle.append(np.random.choice(a=[0, 1], p=oracle(torch.FloatTensor(ob)).detach().numpy()))

            X = pd.DataFrame([ob])
            X.columns = ['o[0]', 'o[1]', 'o[2]', 'o[3]']
            # Addition
            X['o[0]+o[1]'] = X['o[0]'] + X['o[1]']
            X['o[0]+o[2]'] = X['o[0]'] + X['o[2]']
            X['o[0]+o[3]'] = X['o[0]'] + X['o[3]']
            X['o[1]+o[2]'] = X['o[1]'] + X['o[2]']
            X['o[1]+o[3]'] = X['o[1]'] + X['o[3]']
            X['o[2]+o[3]'] = X['o[2]'] + X['o[3]']
            # Multiplication
            X['o[0]*o[0]'] = X['o[0]'] * X['o[0]']
            X['o[1]*o[1]'] = X['o[1]'] * X['o[1]']
            X['o[2]*o[2]'] = X['o[2]'] * X['o[2]']
            X['o[3]*o[3]'] = X['o[3]'] * X['o[3]']
            X['o[0]*o[1]'] = X['o[0]'] * X['o[1]']
            X['o[0]*o[2]'] = X['o[0]'] * X['o[2]']
            X['o[0]*o[3]'] = X['o[0]'] * X['o[3]']
            X['o[1]*o[2]'] = X['o[1]'] * X['o[2]']
            X['o[1]*o[3]'] = X['o[1]'] * X['o[3]']
            X['o[2]*o[3]'] = X['o[2]'] * X['o[3]']

            obs.append(X.to_numpy().tolist()[0])
            #print(X.to_numpy().tolist()[0])
            #exit()

            # student's action (reality)
            action = policy.predict(X)[0]
            #action = policy.predict([ob])[0]

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

    # load Oracle policy
    pi_oracle = torch.load(oracle)
    print(pi_oracle)

    # initial trajectory from oracle
    trajs = pd.read_csv("../Setup/trajectory.csv", nrows=initial_hist)

    X = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']]
    X.columns = ['o[0]', 'o[1]', 'o[2]', 'o[3]']
    # Addition
    X['o[0]+o[1]'] = X['o[0]'] + X['o[1]']
    X['o[0]+o[2]'] = X['o[0]'] + X['o[2]']
    X['o[0]+o[3]'] = X['o[0]'] + X['o[3]']
    X['o[1]+o[2]'] = X['o[1]'] + X['o[2]']
    X['o[1]+o[3]'] = X['o[1]'] + X['o[3]']
    X['o[2]+o[3]'] = X['o[2]'] + X['o[3]']
    # Multiplication
    X['o[0]*o[0]'] = X['o[0]'] * X['o[0]']
    X['o[1]*o[1]'] = X['o[1]'] * X['o[1]']
    X['o[2]*o[2]'] = X['o[2]'] * X['o[2]']
    X['o[3]*o[3]'] = X['o[3]'] * X['o[3]']
    X['o[0]*o[1]'] = X['o[0]'] * X['o[1]']
    X['o[0]*o[2]'] = X['o[0]'] * X['o[2]']
    X['o[0]*o[3]'] = X['o[0]'] * X['o[3]']
    X['o[1]*o[2]'] = X['o[1]'] * X['o[2]']
    X['o[1]*o[3]'] = X['o[1]'] * X['o[3]']
    X['o[2]*o[3]'] = X['o[2]'] * X['o[3]']

    observations = X.to_numpy().tolist()
    actions = trajs['a'].to_numpy().tolist()

    # log times
    start = time.time()

    # initial irl policy
    tree_policy = tree.DecisionTreeClassifier(max_depth=tree_size, random_state=seed)
    tree_policy.fit(observations, actions)

    # evaluate initial
    avg_rew, std_rew = collect_reward(env, tree_policy, 25)
    avg_rewards = [avg_rew]
    std_rewards = [std_rew]


    times = [time.time() - start]

    for i in range(nb_updates):
        print(i)

        # roll out IRL policy, collect imitation data
        obs, act_oracle, rews = dagger_rollout(env, tree_policy, pi_oracle, roll_outs)

        # DAgger style imitation learning (update histories)
        #print(observations)
        #print(obs)
        observations.extend(obs)
        actions.extend(act_oracle)

        # derive IRL policy with updated history
        tree_policy = tree.DecisionTreeClassifier(max_depth=tree_size, random_state=seed)
        tree_policy.fit(observations, actions)

        # evaluate policy
        avg_rew, std_rew = collect_reward(env, tree_policy, 25)
        avg_rewards.append(avg_rew)
        std_rewards.append(std_rew)
        times.append(time.time() - start)

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
    return np.array(avg_rewards), np.array(std_rewards), times


def generate_plots(pomd, oracle, initial_hist, tree_size, nb_updates, roll_outs=1, seed=1):

    i = 0
    rows = 2
    cols = int(np.ceil(len(tree_size) / rows))

    fig, axs = plt.subplots(rows, cols)
    cmap = plt.get_cmap("tab10")

    for r in range(rows):
        for c in range(cols):
            avg_rew, std_rew, times = algo_NDPS(pomd, oracle, initial_hist, tree_size[i], nb_updates)
            # Plot mean and std
            axs[r, c].plot(range(len(avg_rew)), avg_rew, color=cmap(i))
            axs[r, c].fill_between(range(len(avg_rew)), avg_rew - std_rew, avg_rew + std_rew, color=cmap(i), alpha=0.2)
            # Labels and titles
            axs[r, c].set_title("Tree Depth: " + str(tree_size[i]))
            axs[r, c].set_ylim([0, 500])
            axs[r, c].xaxis.set_ticks([j for j in range(0, nb_updates+10, 10)])
            # Solved criterion
            axs[r, c].hlines(475, 0, nb_updates, linestyles="dashed", color='pink')

            i += 1
            if i == len(tree_size):
                break

    fig.suptitle("CartPole-v1: NDPS with Regression Trees")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.set(xlabel='DAgger Rollouts', ylabel='Average Return')
        ax.label_outer()
        ax.xaxis.set_tick_params(which='both', labelbottom=True)

    plt.show()


POMD = "CartPole-v1"
ORACLE = '../Setup/PPO_Sigmoid_2x4_policy.pth'
INIT_HISTORY = 500
TREE_SIZE = range(1, 5)
tree_size = 4
NB_UPDATES = 60

# Plot: Average Reward vs DAgger Rollouts for different tree sizes
generate_plots(POMD, ORACLE, INIT_HISTORY, TREE_SIZE, NB_UPDATES)


# Plot: Average Reward vs Time for 100 Dagger Updates
reward, times = algo_NDPS(POMD, ORACLE, INIT_HISTORY, tree_size, NB_UPDATES)
plt.clf()
plt.plot(times, reward)
plt.title("CartPole-v1: NDPS + DAgger (100 demos) with Regression Tree of Depth " + str(tree_size))
plt.ylim(0, 500)
plt.ylabel("Average Return")
plt.xlabel("Elapsed Time (s)")
plt.tight_layout()
plt.show()
