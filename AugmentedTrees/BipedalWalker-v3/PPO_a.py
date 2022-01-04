import gym
from stable_baselines3 import SAC, DDPG, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.policies import obs_as_tensor

import torch
import os
import pickle
import numpy as np
import copy
from sklearn import tree

import multiprocessing
from itertools import repeat


def get_neurons(obs, relus):
    neurons = []
    for i, relu in enumerate(relus):
        neuron = max(0, np.dot(obs, relu[0]) + relu[1])
        neurons.append(neuron)
    neurons.extend(obs)
    return neurons


def my_program(obs, trees, relus):
    neurons = get_neurons(obs, relus)
    action = []
    for i, tree in enumerate(trees):
        action.append(tree.predict([neurons])[0])
    return action


def train_trees(x, y, depth, seed):
    regr_tree = tree.DecisionTreeRegressor(max_depth=depth, random_state=seed)
    regr_tree.fit(x, y)

    trees = []

    for i in range(4):
        y_i = [item[i] for item in y]
        regr_tree = tree.DecisionTreeRegressor(max_depth=depth, random_state=seed)
        regr_tree.fit(x, y_i)
        trees.append(regr_tree)
    return trees


def save_relus(model, save_to):
    relu_programs = []
    biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        relu_programs.append([w, b])
    pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))
    return relu_programs


def initialize_history(env, model, save_to, games):
    relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))
    observations = []
    actions = []
    neurons = []
    r = 0.0

    for episode in range(games):
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            neurons.append(get_neurons(state, relu_programs))
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    # Save Data
    np.save(file=save_to + 'Observations_' + str(games) + '.npy', arr=observations)
    np.save(file=save_to + 'Actions_' + str(games) + '.npy', arr=actions)
    np.save(file=save_to + 'Neurons_' + str(games) + '.npy', arr=neurons)


def augmented_dagger(env, model, save_to, depth, rollouts, eps_per_rollout, seed):

    X = np.load(save_to + "Neurons_" + str(eps_per_rollout) + ".npy").tolist()
    Y = np.load(save_to + "Actions_" + str(eps_per_rollout) + ".npy").tolist()
    relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))
    best_reward = -10000

    for i in range(rollouts):
        # Regression tree
        trees = train_trees(X, Y, depth, seed)

        # Rollout
        steps = 0
        reward_avg = 0.
        for i in range(eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                # DAGGER
                X.append(get_neurons(ob, relu_programs))
                Y.append(model.predict(ob, deterministic=True)[0])
                # Interact with Environment
                action = my_program(ob, trees, relu_programs)
                ob, r_t, done, _ = env.step(action)
                steps += 1
                reward_avg += r_t
            print(reward_avg / i)

        # 100 consecutive eps
        for i in range(100 - eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                action = my_program(ob, trees, relu_programs)
                ob, r_t, done, _ = env.step(action)
                reward_avg += r_t
        reward_avg = reward_avg / 100.

        # Update best program
        if reward_avg > best_reward:
            best_reward = reward_avg
            best_program = copy.deepcopy(trees)
            print(best_reward)

    return best_reward, best_program


def base_dagger(env, model, save_to, depth, rollouts, eps_per_rollout, seed):

    X = np.load(save_to + "Observations_" + str(eps_per_rollout) + ".npy").tolist()
    Y = np.load(save_to + "Actions_" + str(eps_per_rollout) + ".npy").tolist()
    best_reward = -10000

    for i in range(rollouts):
        # Fit tree
        regr_tree = tree.DecisionTreeRegressor(max_depth=depth, random_state=seed)
        regr_tree.fit(X, Y)

        # Rollout
        steps = 0
        reward_avg = 0
        for i in range(eps_per_rollout):
            ob = env.reset()
            reward = 0.0
            done = False
            while not done:
                # DAGGER
                X.append(ob)
                Y.append(model.predict(ob, deterministic=True)[0])
                # Interact with Environment
                action = regr_tree.predict([ob])[0]
                ob, r_t, done, _ = env.step(action)
                steps += 1
                reward += r_t
            reward_avg += reward
        reward_avg /= eps_per_rollout

        # Update best program
        if reward_avg > best_reward:
            best_reward = reward_avg
            best_program = copy.deepcopy(regr_tree)

    # Evaluate Best Program:
    averaged = 0.0
    for i in range(100):
        ob = env.reset()
        reward = 0.0
        done = False
        while not done:
            action = best_program.predict([ob])[0]
            ob, r_t, done, _ = env.step(action)
            reward += r_t
        averaged += reward
    averaged /= 100.

    return averaged, best_program


def main(seed, l1_actor, l2_actor):

    # configure directory
    save_to = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    print(save_to)
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if l2_actor == 0:
        net_arch = [l1_actor]
    else:
        net_arch = [l1_actor, l2_actor]

    # create environment
    seed = seed
    env = gym.make("BipedalWalker-v3")
    env.seed(seed)

    # training completion requirements
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300., verbose=0)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=0)

    # train oracle
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[256, 256])], activation_fn=torch.nn.ReLU))
    #model.learn(int(1e10), callback=eval_callback)

    # save oracle
    #model.save(save_to + 'model')
    model = model.load(save_to + 'model')
    print("PPO trained")

    # save ReLU programs from actor network
    #save_relus(model, save_to)

    # generate experience
    #initialize_history(env, model, save_to, 25)

    # DAgger using (neuron, action) as training data. Various tree sizes.
    rewards = []
    programs = []
    for depth in range(2, 5):
        reward, program = augmented_dagger(env, model, save_to, depth, 25, 25, seed)
        rewards.append(reward)
        programs.append(program)
        print("Depth: ", depth)
        print("Reward: ", reward)
        print(tree.export_text(program))

    #np.save(file=save_to + 'AugTreeRewards.npy', arr=rewards)
    #pickle.dump(programs, file=open(save_to + 'AugTreePrograms.pkl', "wb"))
    return

    # DAgger using Base DSL.
    rewards = []
    programs = []
    for depth in range(2, 10):
        reward, program = base_dagger(env, model, save_to, depth, 25, 25, seed)
        rewards.append(reward)
        programs.append(program)
        print("Depth: ", depth)
        print("Reward: ", reward)
        print(tree.export_text(program))

    np.save(file=save_to + 'BaseTreeRewards.npy', arr=rewards)
    pickle.dump(programs, file=open(save_to + 'BaseTreePrograms.pkl', "wb"))


if __name__ == "__main__":

    pool = multiprocessing.Pool(10)
    pool.starmap(main, zip(range(1, 11), repeat(128), repeat(128)))
