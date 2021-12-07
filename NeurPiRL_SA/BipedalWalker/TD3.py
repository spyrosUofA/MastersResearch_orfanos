import gym
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import torch
import numpy as np
import os
import pickle

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
    env = gym.make("BipedalWalker-v3")
    env.seed(seed)

    # train oracle
    model = TD3('MlpPolicy', env, verbose=1, policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=net_arch))
    model.learn(int(1e5))

    # save oracle
    model.save(save_to + 'model')
    model = model.load(save_to + 'model')

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
        #main(seed, 64, 64)
        main(seed, 256, 0)


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