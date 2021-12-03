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

    # create environment
    env = gym.make("Pendulum-v0")
    env.seed(seed)

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=220, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    # train oracle
    model = SAC('MlpPolicy', env,
                  #policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[l1_actor, l2_actor], vf=[64, 64])]),
                  seed=seed,
                  batch_size=512,
                  buffer_size=50000,
                  ent_coef=0.1,
                  gamma=0.09999,
                  gradient_steps=32,
                  learning_rate=0.003,
                  tau=0.01,
                  train_freq=32,
                  use_sde=True,
                  verbose=1)


    model = TD3('MlpPolicy', env, verbose=1, policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[128, 128])) #   dict(actor=[l1_actor, l2_actor], vf=[64, 64])]))
    #model.learn(int(1e5)) #, callback=eval_callback)

    # save oracle
    #model.save(save_to + 'model')
    model = model.load(save_to + 'model')

    # save ReLU programs from actor network
    relu_programs = []
    biases = model.policy.state_dict()['actor.mu.0.bias'].detach().numpy()
    weights = model.policy.state_dict()['actor.mu.0.weight'].detach().numpy()
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        print(w, b)
        relu_programs.append([w, b])
    pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))
    print(relu_programs)

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
            print(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    print(r)
            #env.render()
    env.close()
    print(len(observations))
    np.save(file=save_to + 'Observations.npy', arr=observations)
    np.save(file=save_to + 'Actions.npy', arr=actions)


if __name__ == "__main__":
    main(1, 64, 64)

    #for seed in range(1, 16):
    #    main(seed, 2, 4)
    #    main(seed, 64, 64)
