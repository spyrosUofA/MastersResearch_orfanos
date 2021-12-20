import gym
from stable_baselines3 import SAC, DDPG, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.policies import obs_as_tensor

import matplotlib.pyplot as plt

import torch
import os
import pickle
import numpy as np
import copy

def predict_proba(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np


def my_program00(obs):
    n0 = [[ 0.28382367,  1.7934723,  -0.11373619,  1.584884,    0.23984677, -0.42233634, 1.0650071,   2.0341644 ], 0.5301271]
    n1 = [[ 0.36524048, -0.19711488, -0.16623792, -2.815942,   -0.7019816,  -0.32718712, -0.26316372, -0.2660255 ], 0.42466605]
    n2 = [[-1.2129432,   0.37068072, -0.8900809,  -0.8597776,   2.3637958,   2.5205786, -0.40225548, -0.35761788], 0.4344356]
    n3 = [[ 0.38897833,  0.45173603,  1.38248,    -0.85729873, -2.597608 ,  -0.97952, -0.29168594,  0.31902564], 0.40432847]

    n0 = max(0, np.dot(obs, n0[0]) + n0[1])
    n1 = max(0, np.dot(obs, n1[0]) + n1[1])
    n2 = max(0, np.dot(obs, n2[0]) + n2[1])
    n3 = max(0, np.dot(obs, n3[0]) + n3[1])

    if min(n1, n2, n3) < 0.1:
        return 0

    if n0 * 1.21 + 0.31 < -2:
        return 2

    if n2 * -1.63728 + -0.22375107 < -.7:
        return 1

    if n0 * 1.21 + 0.31 > 1:
        return 0


    if n3 * -1.2812959 + -0.16854282 < -1.5:
        return 3

    print("HELLO")
    return 2


def main(seed=0, l1_actor=4, l2_actor=8):
    import numpy as np

    def get_neurons(obs, relus):
        neurons = []
        for i, _ in enumerate(relus):
            neuron = max(0, np.dot(obs, relus[i][0]) + relus[i][1])
            neurons.append(neuron)
        return neurons

    def my_program(obs, tree, relus):
        neurons = get_neurons(obs, relus)
        return tree.predict([neurons])[0]

    # configure directory
    save_to = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if l2_actor == 0:
        net_arch = [l1_actor]
    else:
        net_arch = [l1_actor, l2_actor]

    # create environment
    env = gym.make("LunarLander-v2")
    env.seed(seed)

    # completion requirements
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    # train oracle
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[128, 128])], activation_fn=torch.nn.ReLU))
    #model.learn(int(1e7), callback=eval_callback)

    # save oracle
    #model.save(save_to + 'model')
    model = model.load(save_to + 'model')

    # save ReLU programs from actor network
    save_relus = False
    if save_relus:
        relu_programs = []
        biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
        weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
        for i in range(len(biases)):
            w = weights[i]
            b = biases[i]
            relu_programs.append([w, b])
        pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))

    # generate experience
    generate_exp = True
    if generate_exp:
        relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))
        observations = []
        actions = []
        neurons = []
        r = 0.0
        games = 20

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
        print(r / games)
        env.close()
        np.save(file=save_to + 'Observations_20.npy', arr=observations)
        np.save(file=save_to + 'Actions_20.npy', arr=actions)
        np.save(file=save_to + 'Neurons_20.npy', arr=neurons)

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

    augment_dsl = True
    if augment_dsl:
        print("AUG DSL")

        rollouts = 25
        eps_per_rollout = 25
        render = False
        best_reward = -10000

        from sklearn import tree
        X = np.load(save_to + "Neurons_20.npy").tolist()
        Y = np.load(save_to + "Actions_20.npy").tolist()
        relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))

        for i in range(rollouts):
            # Regression tree for Neuron 1
            regr_1 = tree.DecisionTreeClassifier(max_depth=3, random_state=seed)
            regr_1.fit(X, Y)

            steps = 0
            averaged = 0.0

            for i in range(eps_per_rollout):
                ob = env.reset()
                reward = 0.0
                while True:
                    if render:
                        env.render()
                    action = my_program(ob, regr_1, relu_programs)
                    # DAGGER
                    X.append(get_neurons(ob, relu_programs))
                    Y.append(model.predict(ob, deterministic=True)[0])
                    # Interact with Environment
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

            print(averaged)
            #print(tree.export_text(regr_1)) #, feature_names=['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']))

        averaged = 0.0
        for i in range(100):
            ob = env.reset()
            reward = 0.0
            while True:
                if render:
                    env.render()
                action = my_program(ob, best_program, relu_programs)

                ob, r_t, done, _ = env.step(action)
                reward += r_t

                if done:
                    env.close()
                    break
            averaged += reward

        print("Best Program", averaged / 100)
        print(tree.export_text(best_program))
        print(relu_programs)

    base_dsl = True
    if base_dsl:
        print("BASE DSL")
        import numpy as np
        from sklearn import tree
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


if __name__ == "__main__":

    for seed in range(1, 2):
        #print(seed)
        #main(seed, 32, 0)
        main(seed, 32, 0)
        #main(seed, 4, 0)


        #main(seed, 64, 64)
        #main(seed, 256, 0)


"""
1
-1.2777066645432233
-2.492701652007135
2
-119.85978969935202
-121.3353643321041
3
-232.27027015377652
-229.80714221503638
4
-130.1147025222528
-122.61033272231212
5
-125.29420101444306
-120.88741728080335
6
-117.15158056898993
-117.80184923022763
7
-234.9650876058741
-228.62577243909206
8
-137.81167075717465
-122.32469877351645
9
-131.96307443162453
-243.54067767102578
10
-236.047341486253
-238.90864800735054
11
-127.89789549115983
-126.0870405482137
12
-1.7809709119883814
-2.425385165475506
13
-236.54110828851583
-235.33758862350533
14
-239.28248879296316
-344.5489577225242
15
-126.61432946629326
-123.03310897109166




 # save ReLU programs from actor network
    relu_programs = []
    biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()


    w2 = model.policy.state_dict()['action_net.weight'].detach().numpy()
    b2 = model.policy.state_dict()['action_net.bias'].detach().numpy()

    print("w2 b2", w2, b2)
    ob = env.reset()

    # g(Wx +b)
    neurons = []
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]

        print(i)
        print(w, b)

        neuron = np.maximum(0.0, np.dot(w, ob) + b)
        neurons.append(neuron)  # 8x1
    a0 = np.dot(w2[0], neurons) + b2[0]
    a1 = np.dot(w2[1], neurons) + b2[1]
    a2 = np.dot(w2[2], neurons) + b2[2]
    a3 = np.dot(w2[3], neurons) + b2[3]
    #print("out", a0, a1, a2, a3)

    # output

    plot = False
    if plot:
        observations = []
        actions = []
        r = 0

        a0 = []
        a1 = []
        a2 = []
        a3 = []

        for episode in range(1):
            state = env.reset()
            done = False
            while not done:

                action, _ = model.predict(state, deterministic=True)

                print(action, "vs", my_program(state))
                # Record Trajectory
                observations.append(state)
                actions.append(action)
                # Interact with Environment
                state, reward, done, _ = env.step(action)
                r += reward

                neurons = []
                weighted_neurons = []
                for i in range(len(biases)):
                    w = weights[i]
                    b = biases[i]
                    neuron = max(0.0, np.dot(w, state) + b)
                    weighted_neuron = neuron * (w2[action][i] + b2[action])
                    neurons.append(neuron)  # 32x1
                    weighted_neurons.append(weighted_neuron)
                    relu_programs.append([w, b])

                print(action, neurons, weighted_neurons)

                if action == 0:
                    a0.append([neurons, weighted_neurons])
                elif action == 1:
                    a1.append([neurons, weighted_neurons])
                elif action == 2:
                    a2.append([neurons, weighted_neurons])
                elif action == 3:
                    a3.append([neurons, weighted_neurons])

            data = [np.mean(a0, axis=0), np.mean(a1, axis=0), np.mean(a2, axis=0), np.mean(a3, axis=0)]
            print(data)

            fig = plt.figure()
            gs = fig.add_gridspec(4, hspace=0)
            axs = gs.subplots(sharex=True, sharey=True)
            fig.suptitle('Sharing both axes')
            axs[0].plot(range(4), data[0][0])
            axs[0].plot(range(4), data[0][1])
            axs[1].plot(range(4), data[1][0])
            axs[1].plot(range(4), data[1][1])
            axs[2].plot(range(4), data[2][0])
            axs[2].plot(range(4), data[2][1])
            axs[3].plot(range(4), data[3][0])
            axs[3].plot(range(4), data[3][1])

            plt.show()




    neurons = []
    weighted_neurons = []
    #ob = env.reset()
    #ob = env.reset()
    print(model.predict(ob))
    selected_action, _ = model.predict(ob, deterministic=True)


    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        neuron = max(0.0, np.dot(w, ob) + b)
        weighted_neuron = neuron * w2[selected_action][i] + b2[selected_action]
        neurons.append(neuron)  # 32x1
        weighted_neurons.append(weighted_neuron)
        relu_programs.append([w, b])

    print("L1 Neurons", neurons)
    print("L0", weighted_neurons)

    print(np.dot(w2[0], neurons) + b2[0])
    print(np.dot(w2[1], neurons) + b2[1])
    print(np.dot(w2[2], neurons) + b2[2])
    print(np.dot(w2[3], neurons) + b2[3])



    for episode in range(1):
        state = env.reset()
        done = False
        while not done:

            action, _ = model.predict(state, deterministic=True)
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward

            neurons = []
            weighted_neurons = []
            for i in range(len(biases)):
                w = weights[i]
                b = biases[i]
                neuron = max(0.0, np.dot(w, state) + b)
                weighted_neuron = neuron * (w2[action][i] + b2[action])
                neurons.append(neuron)  # 32x1
                weighted_neurons.append(neuron)
                relu_programs.append([w, b])


            if action == 0:
                a0.append(weighted_neurons)
            elif action == 1:
                a1.append(weighted_neurons)
            elif action == 2:
                a2.append(weighted_neurons)
            elif action == 3:
                a3.append(weighted_neurons)

    data = [np.mean(a0, axis=0),  np.mean(a1, axis=0),  np.mean(a2, axis=0),  np.mean(a3, axis=0)]
    print(data)

    plt.plot(range(8), data[0])
    plt.show()
    plt.clf()
    plt.plot(range(8), data[1])
    plt.show()
    plt.clf()
    plt.plot(range(8), data[2])
    plt.show()
    plt.clf()
    plt.plot(range(8), data[3])
    plt.show()


    print("a0", np.mean(a0, axis=0))
    print("a1", np.mean(a1, axis=0))
    print("a2", np.mean(a2, axis=0))
    print("a3", np.mean(a3, axis=0))

    print(r)



    exit()
    ob = env.reset()
    ob = env.reset()
    ob = env.reset()
    ob = env.reset()

    selected_action, _ = model.predict(ob, deterministic=True)

    neurons = []
    weighted_neurons = []
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        neuron = max(0.0, np.dot(w, ob) + b)
        weighted_neuron = neuron * (w2[selected_action][i] + b2[selected_action])
        neurons.append(neuron)  # 32x1
        weighted_neurons.append(weighted_neuron)
        relu_programs.append([w, b])

    print("L1 Neurons", neurons)
    print("L0", weighted_neurons)

    exit()
    print(np.dot(w2[0], neurons) + b2[0])
    print(np.dot(w2[1], neurons) + b2[1])
    print(np.dot(w2[2], neurons) + b2[2])
    print(np.dot(w2[3], neurons) + b2[3])

    exit()
    pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))

    # save 1 episode rollout
    observations = []
    actions = []
    r = 0
    for episode in range(1):
        state = env.reset()
        done = False
        while not done:
            print(model.policy(state))
            action, _ = model.predict(state, deterministic=True)
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward

            for i in range(len(biases)):
                w = weights[i]
                b = biases[i]
                neuron = max(0.0, np.dot(w, ob) + b)
                weighted_neuron = neuron * (w2[selected_action][i] + b2[selected_action])
                neurons.append(neuron)  # 32x1
                weighted_neurons.append(weighted_neuron)
                relu_programs.append([w, b])

            print("L1 Neurons", neurons)

    print(r)
    env.close()
    np.save(file=save_to + 'Observations.npy', arr=observations)
    np.save(file=save_to + 'Actions.npy', arr=actions)

    # save 20 episode rollout
    observations = []
    actions = []
    r = 0
    for episode in range(20):
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    print(r / 20.0)
    env.close()
    np.save(file=save_to + 'Observations_20.npy', arr=observations)
    np.save(file=save_to + 'Actions_20.npy', arr=actions)



"""