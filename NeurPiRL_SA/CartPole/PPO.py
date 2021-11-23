import gym
from stable_baselines3 import PPO
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
    env = gym.make("CartPole-v1")
    env.seed(seed)

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=490, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    # train oracle
    model = PPO('MlpPolicy', env,
                  policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[l1_actor, l2_actor], vf=[64, 64])]),
                  seed=seed,
                  batch_size=64,
                  ent_coef=0.01,
                  gae_lambda=0.98,
                  gamma=0.999,
                  n_epochs=4,
                  n_steps=1024,
                  verbose=0)
    #model.learn(int(1e10), callback=eval_callback)

    # save oracle
    #model.save(save_to + 'model')
    model = model.load(save_to + 'model')

    # save ReLU programs from actor network
    relu_programs = []
    biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        relu_programs.append([w, b])
    pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))
    print(relu_programs)

    # save 1 episode rollout
    observations = []
    actions = []
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
    env.close()
    print(len(observations))
    np.save(file=save_to + 'Observations.npy', arr=observations)
    np.save(file=save_to + 'Actions.npy', arr=actions)


if __name__ == "__main__":
    for seed in range(1, 16):
        main(seed, 2, 4)
        main(seed, 64, 64)
