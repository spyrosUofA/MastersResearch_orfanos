import gym
import pyglet
import torch as th

from stable_baselines3 import DQN

seed = 0
env = gym.make("LunarLander-v2")
env.seed(seed)

# train oracle
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[64, 64]),
              seed=seed,
              batch_size=64,
              learning_rate=1e-3,
              gamma=0.999,
              verbose=1)
model.learn(2612224)

# save oracle
model.save("Oracle/Oracle-" + str(seed) + '/model')



# save relus
model = PPO.load("Oracle/Oracle-" + str(seed) + '/model')
while True:
  done = False
  obs = env.reset()
  ep_rew = 0
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    ep_rew += reward
    env.render()
  print(ep_rew)


