import gym
import keras_gym as km
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn import tree

###############################################################################
# environment (MDP)
###############################################################################

env = gym.make('Pendulum-v0')
env = km.wrappers.BoxActionsToReals(env)
env = km.wrappers.TrainMonitor(
    env=env, tensorboard_dir='/tmp/tensorboard/pendulum/ppo_static')
km.enable_logging()


###############################################################################
# function approximator
###############################################################################

class MLP(km.FunctionApproximator):
    def body(self, X):
        X = keras.layers.Lambda(
            lambda x: K.concatenate([x, K.square(x)], axis=1))(X)
        X = keras.layers.Dense(units=32, activation='relu')(X)
        X = keras.layers.Dense(units=32, activation='relu')(X)
        return X


mlp = MLP(env, lr=5e-3)
pi = km.GaussianPolicy(mlp, update_strategy='ppo')
v = km.V(mlp, gamma=0.99, bootstrap_n=5)
ac = km.ActorCritic(pi, v)


buffer = km.caching.ExperienceReplayBuffer.from_value_function(
    value_function=v, capacity=1000, batch_size=32)


def my_program(obs, tree):
    if tree is None:
        return 0
    return tree.predict([obs])[0]


def evaluate_program(tree, nb_eps=50):
    averaged = 0.0
    for i in range(nb_eps):
        ob = env.reset()
        reward = 0.0
        done = False
        while not done:
            action = my_program(ob, tree)
            ob, r_t, done, _ = env.step([action])
            reward += r_t
        averaged += reward
    averaged /= nb_eps
    return averaged


reg_tree = None
lam_mix = 0.75
mixed_actions = []
states = []

###############################################################################
# run
###############################################################################

while env.T < 1000000:
    s = env.reset()
    for t in range(env.spec.max_episode_steps):
        a = pi(s, use_target_model=True)
        a_mixed = a * lam_mix + (1.0-lam_mix) * my_program(s, reg_tree)

        states.append(s)
        mixed_actions.append(a_mixed)

        s_next, r, done, info = env.step(a_mixed)

        buffer.add(s, a, r, done, env.ep)
        if len(buffer) >= buffer.capacity:

            # update tree
            reg_tree = tree.DecisionTreeRegressor(max_depth=5)
            reg_tree.fit(states, mixed_actions)
            print(evaluate_program(reg_tree, 50))
            mixed_actions = []
            states = []

            # use 8 epochs per round
            num_batches = int(8 * buffer.capacity / buffer.batch_size)
            for _ in range(num_batches):
                ac.batch_update(*buffer.sample())
            buffer.clear()
            pi.sync_target_model(tau=0.1)

        if done:
            break

        s = s_next
