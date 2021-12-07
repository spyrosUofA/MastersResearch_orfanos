import gym

def collect_reward(nb_episodes, render=False):
    env = gym.make("Pendulum-v0")


    steps = 0
    averaged = 0.0
    for i in range(nb_episodes):
        ob = env.reset()
        reward = 0.0
        while True:
            if render and i == 15:
                env.render()
            action = [-2 * ob[1] - (8 * ob[1] + 2 * ob[2]) / ob[0]] # PENDULUM

            ob, r_t, done, _ = env.step(action)
            steps += 1
            reward += r_t

            if done:
                env.close()
                break
        averaged += reward

    return averaged / nb_episodes


print(collect_reward(200, True))