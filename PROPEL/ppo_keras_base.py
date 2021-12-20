import gym
import keras_gym as km
from tensorflow import keras
from tensorflow.keras import backend as K


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
        X = keras.layers.Dense(units=6, activation='tanh')(X)
        X = keras.layers.Dense(units=6, activation='tanh')(X)
        return X


mlp = MLP(env, lr=1e-3)
pi = km.GaussianPolicy(mlp, update_strategy='ppo')
v = km.V(mlp, gamma=0.9, bootstrap_n=5)
ac = km.ActorCritic(pi, v)


buffer = km.caching.ExperienceReplayBuffer.from_value_function(
    value_function=v, capacity=512, batch_size=32)


###############################################################################
# run
###############################################################################

while env.T < 1000000:
    s = env.reset()
    for t in range(env.spec.max_episode_steps):
        a = pi(s, use_target_model=True)
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, env.ep)
        if len(buffer) >= buffer.capacity:
            # use 4 epochs per round
            num_batches = int(4 * buffer.capacity / buffer.batch_size)
            for _ in range(num_batches):
                ac.batch_update(*buffer.sample())
            buffer.clear()
            pi.sync_target_model(tau=0.1)

        if done:
            break

        s = s_next
