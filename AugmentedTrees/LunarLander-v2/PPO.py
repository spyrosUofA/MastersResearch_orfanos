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
    for i, _ in enumerate(relus):
        neuron = max(0, np.dot(obs, relus[i][0]) + relus[i][1])
        neurons.append(neuron)
    return neurons


def my_program(obs, tree, relus):
    neurons = get_neurons(obs, relus)
    return tree.predict([neurons])[0]


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
        regr_tree = tree.DecisionTreeClassifier(max_depth=depth, random_state=seed)
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
                X.append(get_neurons(ob, relu_programs))
                Y.append(model.predict(ob, deterministic=True)[0])
                # Interact with Environment
                action = my_program(ob, regr_tree, relu_programs)
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
            action = my_program(ob, best_program, relu_programs)
            ob, r_t, done, _ = env.step(action)
            reward += r_t
        averaged += reward
    averaged /= 100.

    return averaged, best_program


def base_dagger(env, model, save_to, depth, rollouts, eps_per_rollout, seed):

    X = np.load(save_to + "Observations_" + str(eps_per_rollout) + ".npy").tolist()
    Y = np.load(save_to + "Actions_" + str(eps_per_rollout) + ".npy").tolist()
    best_reward = -10000

    for i in range(rollouts):
        # Fit tree
        regr_tree = tree.DecisionTreeClassifier(max_depth=depth, random_state=seed)
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
    if not os.path.exists(save_to):
        print(save_to)
        os.makedirs(save_to)

    if l2_actor == 0:
        net_arch = [l1_actor]
    else:
        net_arch = [l1_actor, l2_actor]

    # create environment
    seed = seed
    env = gym.make("LunarLander-v2")
    env.seed(seed)

    # training completion requirements
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=240, verbose=0)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=0)

    # train oracle
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[128, 128])], activation_fn=torch.nn.ReLU))
    #model.learn(int(1e10), callback=eval_callback)

    # save oracle
    #model.save(save_to + 'model')
    model = model.load(save_to + 'model')

    # save ReLU programs from actor network
    save_relus(model, save_to)

    # generate experience
    initialize_history(env, model, save_to, 25)

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

    np.save(file=save_to + 'AugTreeRewards.npy', arr=rewards)
    pickle.dump(programs, file=open(save_to + 'AugTreePrograms.pkl', "wb"))
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

    names = []
    relu_prog = []
    for i, _ in enumerate(l1_actor):
        name = '(' + str(np.around(relu_prog[0][0], 3)) + ' *dot* obs[:] + ' + str(np.round(relu_prog[0][1], 3)) + ')'
        names.append(name)


    base_dsl = False
    if base_dsl:
        print("BASE DSL")
        X = np.load(save_to + "Observations_20.npy").tolist()
        Y = np.load(save_to + "Actions_20.npy").tolist()

        rollouts = 25
        eps_per_rollout = 25
        render = False
        best_reward = -1000

        for i in range(rollouts):
            # Regression tree for Neuron 1
            regr_1 = tree.DecisionTreeClassifier(max_depth=15, random_state=seed)
            regr_1.fit(X, Y)

            steps = 0
            averaged = 0.0

            for i in range(eps_per_rollout):
                ob = env.reset()
                reward = 0.0
                while True:
                    if render:
                        env.render()
                    action = regr_1.predict([ob])[0]

                    # DAGGER
                    X.append(ob)
                    Y.append(model.predict(ob, deterministic=True)[0])

                    ob, r_t, done, _ = env.step(action)
                    steps += 1
                    reward += r_t

                    if done:
                        env.close()
                        break
                averaged += reward
            averaged /= eps_per_rollout

            if averaged > best_reward:
                best_reward = averaged
                best_program = copy.deepcopy(regr_1)

        print(best_reward)
        print(tree.export_text(best_program))

    plot = False
    if plot:

        observations = []
        actions = []
        r = 0

        a0 = []
        a1 = []
        a2 = []
        a3 = []

        biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
        weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()

        w2 = model.policy.state_dict()['action_net.weight'].detach().numpy()
        b2 = model.policy.state_dict()['action_net.bias'].detach().numpy()

        export_dataN = []
        export_dataA = []

        for episode in range(20):
            state = env.reset()
            done = False
            while not done:

                action, _ = model.predict(state, deterministic=True)

               # print(action, "vs", my_program(state))
                # Record Trajectory
                observations.append(state)
                actions.append(action)
                # Interact with Environment
                state, reward, done, _ = env.step(action)
                r += reward

                neurons = []
                weighted_neuron = 0
                for i in range(len(biases)):
                    w = weights[i]
                    b = biases[i]
                    neuron = max(0.0, np.dot(w, state) + b)
                    neurons.append(neuron)

                    weighted_neuron += neuron * w2[action][i]
                weighted_neuron += b2[action]

                logits = []
                for i, _ in enumerate(b2):
                    logit = np.dot(w2[i], neurons) + b2[i]
                    logits.append(logit)

                weighted_neuron = logits
                print(action, neurons, logits, np.nanargmax(logits))
                obs = obs_as_tensor(state, model.policy.device)

                export_dataN.append(neurons)
                export_dataA.append(action)

                if action == 0:
                    a0.append([neurons, weighted_neuron])
                elif action == 1:
                    a1.append([neurons, weighted_neuron])
                elif action == 2:
                    a2.append([neurons, weighted_neuron])
                elif action == 3:
                    a3.append([neurons, weighted_neuron])


        np.save(file=save_to + 'neurons.npy', arr=export_dataN)
        np.save(file=save_to + 'neurons_to_actions.npy', arr=export_dataA)


        data = [np.mean(a0, axis=0), np.mean(a1, axis=0), np.mean(a2, axis=0), np.mean(a3, axis=0)]
        print(data)

        fig = plt.figure()
        gs = fig.add_gridspec(l1_actor, hspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        fig.suptitle('Sharing both axes')

        for i in range(4):
            axs[i].plot(range(i), data[i][0])
            axs[i].plot(range(i), data[i][1])

        plt.show()


if __name__ == "__main__":

    pool = multiprocessing.Pool(8)
    pool.starmap(main, zip(range(16, 30), repeat(4), repeat(0)))
    pool.starmap(main, zip(range(16, 30), repeat(32), repeat(0)))
    pool.starmap(main, zip(range(16, 30), repeat(64), repeat(64)))
    pool.starmap(main, zip(range(16, 30), repeat(256), repeat(256)))

    #for seed in range(2, 16):
    #    main(seed, 4, 0)
    #    main(seed, 32, 0)
    #    main(seed, 64, 64)
    #    main(seed, 256, 256)
