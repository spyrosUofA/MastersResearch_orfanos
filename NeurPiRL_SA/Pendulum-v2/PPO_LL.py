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
from sklearn import tree




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
    np.save(file=save_to + 'Observations_' + str(games) + ' .npy', arr=observations)
    np.save(file=save_to + 'Actions_' + str(games) + '.npy', arr=actions)
    np.save(file=save_to + 'Neurons_' + str(games) + '.npy', arr=neurons)


def augmented_dagger(env, model, save_to, depth, rollouts, eps_per_rollout):

    X = np.load(save_to + "Neurons_" + str(eps_per_rollout) + ".npy").tolist()
    Y = np.load(save_to + "Actions_" + str(eps_per_rollout) + ".npy").tolist()
    relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))
    best_reward = -10000

    for i in range(rollouts):
        # Regression tree
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
            print(best_reward)
            print(tree.export_text(best_program))

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


def base_dagger(env, model, save_to, depth, rollouts, eps_per_rollout):

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





def main(seed=0, l1_actor=4, l2_actor=8):
    # configure directory
    save_to = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if l2_actor == 0:
        net_arch = [l1_actor]
    else:
        net_arch = [l1_actor, l2_actor]

    # create environment
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)

    #
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=240, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    # train oracle
    model = PPO('MlpPolicy', env,
                  policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=net_arch, vf=[128, 128])]),
                  seed=seed,
                  batch_size=64,
                  ent_coef=0.01,
                  gae_lambda=0.98,
                  gamma=0.999,
                  n_epochs=4,
                  n_steps=1024,
                  verbose=0)

    #model.learn(int(1e10), callback=eval_callback)

    # train oracle
    #model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=net_arch))

    # save oracle
    #model.save(save_to + 'model')
    model = model.load(save_to + 'model')

    #save_relus(model, save_to)

    # generate experience
    #initialize_history(env, model, save_to, 25)

    # DAgger using (neuron, action) as training data. Various tree sizes.
    rewards = []
    programs = []
    for depth in range(2, 4):
        reward, program = augmented_dagger(env, model, save_to, depth, 50, 25)
        rewards.append(reward)
        programs.append(program)
        print("Depth: ", depth)
        print("Reward: ", reward)
        print(tree.export_text(program))
    exit("DONE")


    ob = env.reset()

    # From Input Layer to 1st hidden layer (8 inputs to 1 neuron)
    w1 = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1 = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()

    # max(0, W1x +b1)
    neurons = []
    for i, _ in enumerate(b1):
        neuron = np.maximum(0.0, np.dot(w1[i], ob) + b1[i])
        neurons.append(neuron)  # layer_size x 1

    # Now we are at the first hidden layer. One more layer to go!
    w2 = model.policy.state_dict()['action_net.weight'].detach().numpy()
    b2 = model.policy.state_dict()['action_net.bias'].detach().numpy()

    log_probs = []  # ?
    for i, _ in enumerate(b2):
        log_prob = np.dot(w2[i], neurons) + b2[i]
        log_probs.append(log_prob)
    print(log_probs)
    print(model.predict(ob, deterministic=True))
    exit()


    biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()

    ob = env.reset()
    weighted_neurons = []
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        neuron = max(0.0, np.dot(w, ob) + b)
        print(neuron)
    exit()

    # save ReLU programs from actor network
    relu_programs = []
    biases = model.policy.state_dict()['actor.mu.0.bias'].detach().numpy()
    weights = model.policy.state_dict()['actor.mu.0.weight'].detach().numpy()
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        relu_programs.append([w, b])
    pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))

    # save 1 episode rollout
    observations = []
    actions = []
    r = 0
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
    print(r)

    env.close()
    np.save(file=save_to + 'Observations.npy', arr=observations)
    np.save(file=save_to + 'Actions.npy', arr=actions)


if __name__ == "__main__":

    for seed in range(1, 2):
        print(seed)
        main(seed, 8, 0)
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
"""