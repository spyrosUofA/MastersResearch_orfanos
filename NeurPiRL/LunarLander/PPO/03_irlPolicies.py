import sys
import numpy as np
import matplotlib as mpl

mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    # Task setup block starts
    # Do not change
    env = gym.make('CartPole-v1')
    env.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    # Task setup block end

    # Learner setup block
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Experiment block starts
    ret = 0
    rets = []
    avgrets = []
    o = env.reset()

    def neuron1_policy(o):
        # Split 1
        if o[3] < 0.1643993:
            # Split 2
            if o[0] >= 0.1943893:
                # Split 3
                if o[3] < -0.04705893:
                    return 0  # a = 0
                else:
                    return 1
            # Split 2
            else: # o[0] < 0.1943893
                # Split 3
                if o[3] < -0.1431703:
                    return 0
                else:
                    return 1
        # Split 1
        else:
            # Split 2
            if o[0] >= 0.2381055:
                # Split 3
                if o[3] < -0.04705893:
                    return 0  # a = 0
                else:
                    return 1
            # Split 2
            else:  # o[0] < 0.23810550.5648384
                # Split 3
                if o[3] < -0.1431703:
                    return 0
                else:
                    return 1

    def neuron1a_policy(o):
        if o[3] < 0.1643993:
            if o[0] < 0.1943893:
                return 0  # a = 0
            else:
                return 0  # a = 0
        else:
            if o[0] < 0.2381055:
                return 1  # a = 0 # decision rule had equal split. With a=0 get great score. With a=1 get ~135 score.
            else:
                return 1  # a = 1

    # Policy derived from structure of neuron 2's regression tree
    def neuron2r_policy(o):
        if o[3] < -0.04620776:
            if o[2] < -0.008608433:
                return 0  # a=0
            else:
                return 0  # a=0
        else:
            if o[2] < -0.0219456:
                return 0  # a=0 # decision rule had equal split. With a=0 get great score. With a=1 get ~135 score.
            else:
                return 1  # a=1

    def neuron1python_policy(X):
        # v <= -0.17
        if X[1] <= -0.168:
            # x <= -0.37
            if X[0] <= -0.372:
                # omega <= 0.70
                if X[3] <= 0.7:
                    return 1  # a=0
                # omega > 0.70
                else:
                    return 1
            # x > -0.37
            else:
                # x <= 0.27
                if X[0] <= 0.27:
                    return 1
                # x > 0.27
                else:
                    return 1
        # v > -0.17
        else:
            # x <= -0.33
            if X[0] <= -0.331:
                # v <= 0.03
                if X[1] <= 0.03:
                    return 0
                else:
                    return 0
            # x > -0.33
            else:
                # v <= 0.03
                if X[1] <= -0.04:
                    return 1
                else:
                    return 0

    def neuron1python_policy(X):
        # x <= 0.14
        if X[0] <= 0.14:
            # omega <= 0.24
            if X[3] <= 0.24:
                # omega <= 0.70
                if X[3] <= 0.7:
                    return 1  # a=0
                # omega > 0.70
                else:
                    return 1
            # x > -0.37
            else:
                # x <= 0.27
                if X[0] <= 0.27:
                    return 1
                # x > 0.27
                else:
                    return 1
        # v > -0.17
        else:
            # x <= -0.33
            if X[0] <= -0.331:
                # v <= 0.03
                if X[1] <= 0.03:
                    return 0
                else:
                    return 0
            # x > -0.33
            else:
                # v <= 0.03
                if X[1] <= -0.04:
                    return 1
                else:
                    return 0

    def neuron2relu_policy(X):
        x = X[0]
        v = X[1]
        theta = X[2]
        omega = X[3]

        if omega <= 0.24:
            if theta <= 0.05:
                if theta <= 0.03:
                    return 0 #0
                if theta > 0.03:
                    return 1 # 1 better than 0
            if theta > 0.05:
                if x <= 0.10:
                    return 1 #same
                if x > 0.10:
                    return 1 #same
        if omega > 0.24:
            if theta <= 0.00:
                if x <= 0.09:
                    return 1 #1
                if x > 0.09:
                    return 1 #either
            if theta > 0.00:
                if theta <= 0.03:
                    return 0
                if theta > 0.03:
                    return 1 # big difference if 0

    def imitate_ppo(X):
        x = X[0]
        v = X[1]
        theta = X[2]
        omega = X[3]

        if omega <= -0.06:
            if theta <= 0.04:
                if omega <= -0.31:
                    return 0 #[0.11]
                if omega > -0.31:
                    return 0 #[0.27]
            if theta > 0.04:
                if v <= 0.22:
                    return 1 #[1.00]
                if v > 0.22:
                    return 1 #[0.54]
        if omega > -0.06:
            if omega <= 0.20:
                if theta <= 0.01:
                    return 0 #[0.46]
                if theta > 0.01:
                    return 1 #[0.76]
            if omega > 0.20:
                if theta <= -0.04:
                    return 0 #[0.59]
                if theta > -0.04:
                    return 1 #[0.89]

    num_steps = 500000
    checkpoint = 10000
    for steps in range(num_steps):

        # Select an action
        a = imitate_ppo(o)

        # Observe, update environment
        op, r, done, infos = env.step(a)
        o = op

        # Log
        ret += r
        if done:
            rets.append(ret)
            ret = 0
            o = env.reset()

        if (steps + 1) % checkpoint == 0:
            avgrets.append(np.mean(rets))
            rets = []
            plt.clf()
            plt.ylim(0, 500)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            plt.pause(0.001)

    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(avgrets)))
    data[0] = range(checkpoint, num_steps + 1, checkpoint)
    data[1] = avgrets
    #plt.show()

    plt.clf()
    plt.ylim(0, 500)
    plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
    plt.title("Cartpole-v1 with IRL Policy")
    plt.ylabel("Average Return")
    plt.xlabel("Timestep")
    plt.savefig("IRL.png")
    plt.show()


if __name__ == "__main__":
    main()
