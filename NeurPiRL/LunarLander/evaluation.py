
class Evaluate():

    def __init__(self, nb_evaluations):
        self.nb_evaluations = nb_evaluations

    def eval(self, p):
        steps = 0
        averaged = 0.0

        import gym
        env = gym.make('LunarLander-v2')

        for _ in range(self.nb_evaluations):
            ob = env.reset()
            reward = 0.0
            while True:
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                ob, r_t, done, _ = env.step(action[0])
                steps += 1
                reward += r_t

                if done: break
            averaged += reward

        return averaged / self.nb_evaluations, _, self.nb_evaluations

    def eval_triage(self, p, episode_count):
        steps = 0
        averaged = 0

        import gym
        env = gym.make('LunarLander-v2')

        for _ in range(episode_count):
            ob = env.reset()
            reward = 0
            while True:
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                ob, r_t, done, _ = env.step(action[0])
                steps += 1
                reward += r_t

                if done: break
            averaged += reward

        return averaged / episode_count, _, episode_count