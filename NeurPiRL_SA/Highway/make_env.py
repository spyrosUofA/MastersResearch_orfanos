import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make('highway-v0')
env.seed(0)


# y=0. is lane 1 (upper most lane)
# y=0.25 is lane 2
# y=0.50 is lane 3
# y=0.75 is lane 4

# After changing lanes in takes about 15 simulation frames (source: https://highway-env.readthedocs.io/en/latest/faq.html)


# vehicle starts in top lane (i.e. y = 0.)
obs = env.reset()
env.render()
print(obs[0][2], obs[0][4])  # in lane 1, y velocity 0
print(obs[0][2], obs[1][2], obs[2][2], obs[3][2], obs[4][2])

plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)

# Move down
obs, reward, done, info = env.step(2)
print(obs[0][2], obs[0][4])  # almost in lane 2, moving towards it
plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)


print(obs[0][2], obs[1][2], obs[2][2],obs[3][2], obs[4][2])
exit()

# Wait 15 frames
for _ in range(15):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    print(obs[0][2], obs[0][4])  # in lane 2, y velocity is 0
plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)


# Move down
obs, reward, done, info = env.step(2)
print(obs[0][2], obs[0][4])  # almost in lane 2, moving towards it
plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)

# Wait 15 frames
for _ in range(15):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
print(obs[0][2], obs[0][4])  # in lane 2, y velocity is 0
plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)


# Move up
obs, reward, done, info = env.step(0)
print(obs[0][2], obs[0][4])  # almost in lane 2, moving towards it
plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)

# Wait 15 frames
for _ in range(15):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
print(obs[0][2], obs[0][4])  # in lane 2, y velocity is 0
plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)

exit()



# Nove down
env.render()

obs, reward, done, info = env.step(0)
print(obs)
env.render()


# Wait 15 frames
for _ in range(15):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
print(obs[0][2])  # lane 3
plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)

exit()

env.render()

obs, reward, done, info = env.step(0)
print(obs)
env.render()

exit()


env = gym.make('highway-v0')
env.seed(0)

s0 = env.reset()
print(s0)
env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.pause(5)
plt.clf()

for _ in range(3):
    action = 2 #env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
    print(obs)
    #break
    #env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.pause(20)
exit()