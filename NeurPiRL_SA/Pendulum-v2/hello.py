import gym
import numpy as np

def collect_reward(nb_episodes, render=False):
    env = gym.make("Pendulum-v0")
    env = gym.make("LunarLanderContinuous-v2")

    steps = 0
    averaged = 0.0
    for i in range(nb_episodes):
        ob = env.reset()
        reward = 0.0
        while True:
            if render and i == 15:
                env.render()

            #action = [-2 * ob[1] - (8 * ob[1] + 2 * ob[2]) / ob[0]] # PENDULUM
            action = [-10 * ob[1] + np.sin(ob[2]) - 14 * ob[3] - 1.99, -5.79 * ob[3] / (ob[5] - ob[2])] # LL

            ob, r_t, done, _ = env.step(action)
            steps += 1
            reward += r_t

            if done:
                env.close()
                break
        averaged += reward

    return averaged / nb_episodes


print(collect_reward(200, False))