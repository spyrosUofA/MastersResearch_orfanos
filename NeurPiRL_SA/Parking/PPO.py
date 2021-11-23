import gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3 import HER, SAC, DDPG
#from stable_baselines3.ddpg import NormalActionNoise

env = gym.make("parking-v0")
model = SAC('MlpPolicy', env,
            verbose=1)

model.learn(int(2e4))


model = PPO('MlpPolicy', env,
              #policy_kwargs=dict(net_arch=[256, 256]),
              seed=0,
              batch_size=64,
              ent_coef=0.01,
              gae_lambda=0.98,
              gamma=0.999,
              n_epochs=4,
              n_steps=1024,
              tensorboard_log=None)
model.learn(int(2e4))
model.save("PPO/model")

# Load and test saved model
model = PPO.load("PPO/model")
while True:
  done = False
  obs = env.reset()
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()