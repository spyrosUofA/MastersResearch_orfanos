import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-config', action='store', dest='config', default="E010")
parser.add_argument('-hparam', action='store', dest='hparam', default="sa_cpus-16_n-25_c-None")
parser.add_argument('-start', action='store', dest='start', default=1)
parser.add_argument('-end', action='store', dest='end', default=15)
parameters = parser.parse_args()

NB_RESETS = 15
rewards = []
scores = []




for r in range(1, NB_RESETS +1, 4):
    file = "./logs/" + parameters.config + "/" + parameters.hparam + "_run-" +  str(r)

    with open(file) as f:
        output = [np.array((line.strip()).split(', '), dtype=float).tolist() for line in f]

    scores.append(output[-1][2])
    rewards.append(output[-1][1])

rew_avg = np.mean(rewards, axis=0)
rew_std = np.std(rewards, axis=0) * (len(rewards) ** -0.5)

score_avg = np.mean(scores, axis=0)
score_std = np.std(scores, axis=0) * (len(rewards) ** -0.5)

print("Configuration: ", parameters.config,  " | Average Reward: ", rew_avg, "(+/-)", rew_std, " |  Average Score: ", score_avg, "(+/-)", score_std)
