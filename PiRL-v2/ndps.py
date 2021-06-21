import gym
import torch
import gym
from DSL import Ite, Lt, Observation, Num, AssignAction, Addition, Multiplication
import numpy as np
import copy
from Optimization_so import ParameterFinder
import pandas as pd
import pickle
import time

import matplotlib.pyplot as plt
import matplotlib as mpl

def get_action(obs, p):
    actions = []
    for ob in obs:
        namespace = {'obs': ob, 'act': 0}
        p.interpret(namespace)
        actions.append(namespace['act'])
        # actions.append(namespace['act'].value)  # EDITED...
    return actions


class ProgramList():

    def __init__(self):
        self.plist = {}

    def insert(self, program):

        if program.getSize() not in self.plist:
            self.plist[program.getSize()] = {}

        if program.name() not in self.plist[program.getSize()]:
            self.plist[program.getSize()][program.name()] = []

        self.plist[program.getSize()][program.name()].append(program)

    def init_plist(self, constant_values, observation_values, action_values, boolean_programs):
        for i in observation_values:
            p = Observation(i)
            self.insert(p)

        for i in constant_values:
            p = Num(i)
            self.insert(p)

        for i in action_values:
            p = AssignAction(Num(i))
            self.insert(p)

        for i in range(len(boolean_programs)):
            self.insert(boolean_programs[i])

    def get_programs(self, size):

        if size in self.plist:
            return self.plist[size]
        return {}


class BottomUpSearch():

    def get_histories(self, p, oracle, timesteps):
        steps = 0
        averaged = 0

        env = gym.make('CartPole-v1')

        obs = []
        actions_irl = []
        actions_oracle = []

        ob = env.reset()

        for _ in range(timesteps):
            reward = 0
            obs.append(ob)
            actions_oracle.append(np.random.choice(a=[0, 1], p=oracle(torch.FloatTensor(ob)).detach().numpy()))

            namespace = {'obs': ob, 'act': 0}
            p.interpret(namespace)
            action = [namespace['act']]
            ob, r_t, done, _ = env.step(action[0])

            actions_irl.append(action[0])

            steps += 1
            reward += r_t

            if done:
                break

        return np.array(obs).tolist(), actions_irl, actions_oracle

    def evaluate(self, p, episode_count):
        steps = 0
        averaged = 0

        env = gym.make('CartPole-v1')

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

        return averaged / episode_count

    def init_env(self, inout):
        env = {}
        for v in self._variables:
            env[v] = inout[v]
        return env

    def has_equivalent(self, program, observations, actions):
        p_out = get_action(obs=observations, p=program)
        tuple_out = tuple(p_out)

        if tuple_out not in self.outputs:
            self.outputs.add(tuple_out)
            return False
        return True

    def grow(self, plist, closed_list, operations, size):
        new_programs = []
        for op in operations:
            for p in op.grow(plist, size):
                if p not in closed_list:
                    closed_list.append(p)
                    new_programs.append(p)
                    yield p

        for p in new_programs:
            plist.insert(p)

    def synthesize(self, bound, operations, constant_values, observation_values, action_values, observations, actions,
                   boolean_programs, PiRL=False):

        start = time.time()
        elapsed_times = []
        current_evaluation = []

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, action_values, boolean_programs)

        best_reward = 0
        best_rewards = []
        best_policy = None
        number_evaluations = 0

        self.outputs = set()
        filename = "programs_NPDS"
        if PiRL:
            filename += "_PiRL.txt"
        else:
            filename += ".txt"
        with open(filename, "w") as text_file:
            text_file.write("Best programs:\n")
        parameter_finder = ParameterFinder(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                # print(p.name)
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        # print("test")
                        #print(p_copy.toString())
                        reward = parameter_finder.optimize(p_copy)
                        # print(reward)
                        #print(p_copy.toString())
                    number_evaluations += 1
                    reward = self.evaluate(p_copy, 10)
                    if reward > best_reward:
                        reward = self.evaluate(p_copy, 100)
                        if reward > best_reward:
                            print(p_copy.toString(), reward)
                            with open(filename, "a") as text_file:
                                text_file.write(p_copy.toString() + str(reward) + "\n")
                            best_reward = reward
                            best_policy = p_copy

                            # Recording results
                            best_rewards.append(best_reward)
                            elapsed_times.append(time.time() - start)
                            current_evaluation.append(number_evaluations)

                            plt.clf()
                            plt.step(elapsed_times, best_rewards, where='post')
                            plt.ylim(0, 500)
                            plt.title("CartPole-v1")
                            plt.ylabel("Average Return")
                            plt.xlabel("Elapsed Time (s)")
                            plt.pause(0.01)

                    # """
                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        if best_policy is not None:
            reward = self.evaluate(best_policy, 1000)
            print(best_policy.toString(), reward)
            with open(filename, "a") as text_file:
                text_file.write("best: " + best_policy.toString() + str(reward) + "\n")

        return best_policy, number_evaluations

def algo_NDPS(pi_oracle, sketch, pomd, seed=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)

    # Task setup
    env = gym.make(pomd)
    env.seed(seed)
    obs_space = np.arange(env.observation_space.shape[0])
    action_space = np.arange(env.action_space.n)

    # load Oracle policy
    policy = torch.load(pi_oracle)

    # initial trajectory from oracle
    trajs = pd.read_csv("../Setup/trajectory.csv", nrows=500)
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy().tolist()
    actions = trajs['a'].to_numpy().tolist()

    # initial irl policy
    synthesizer = BottomUpSearch()
    p, num = synthesizer.synthesize(10, sketch, [0.25, 0.0, -0.25], obs_space, action_space, observations,
                                    actions, [], PiRL=True)

    M = 5

    for i in range(M):
        print("k=", str(i))

        # roll out IRL policy, collect imitation data
        obs, act_irl, act_oracle = synthesizer.get_histories(p, policy, 500)

        # DAgger style imitation learning (update histories)
        observations.extend(obs)
        actions.extend(act_oracle)

        # derive IRL policy from program synthesis
        p, num = synthesizer.synthesize(10, sketch, [0.25, 0.0, -0.25], obs_space, action_space, observations,
                                    actions, [], PiRL=True)

    return p

if __name__ == '__main__':
    algo_NDPS(pi_oracle='../Setup/ppo_2x4_policy.pth', pomd='CartPole-v1', sketch=[Ite, Lt], seed=1)

    ## MountainCarContinuous-v0
    # observations = np.load("observations_con.npy")[:100]
    # actions = np.load("actions_con.npy")[:100]

    ## CartPole-v1
    trajs = pd.read_csv("../Setup/trajectory.csv")
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()

