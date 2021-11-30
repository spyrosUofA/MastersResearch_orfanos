import gym
from DSL import Ite, Lt, Observation, Num, AssignAction, Addition, Multiplication, ReLU
import numpy as np
import copy
from OptimizationDiscrete import ParameterFinderDiscrete
from OptimizationContinuous import ParameterFinderContinuous

import pandas as pd
import pickle
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib



def get_action(obs, p):
    actions = []
    for ob in obs:
        namespace = {'obs': ob, 'act': 0}
        p.interpret(namespace)
        actions.append(namespace['act'])
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
    
    def evaluate(self, p, episode_count, env_name='CartPole-v1'):
        steps = 0
        averaged = 0
        
        env = gym.make(env_name)
        
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

    def synthesize(self, bound, operations, constant_values, observation_values, action_values, observations, actions, boolean_programs, name, PiRL = False):

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
        filename = "programs" + name
        graph_name = "eval_time" + name
        if PiRL:
            filename+="_PiRL.txt"
            graph_name += "_PiRL.png"
        else:
            filename+=".txt"
            graph_name += ".png"

        #with open(filename, "w") as text_file:
        #    text_file.write("Best programs:\n")

        parameter_finder = ParameterFinderDiscrete(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        t000 = time.time()
                        score = parameter_finder.optimize(p_copy)
                        #print(time.time() - t000)
                    number_evaluations += 1
                    reward = self.evaluate(p_copy, 10)
                    if reward > best_reward:
                        reward = self.evaluate(p_copy, 100)
                        if reward > best_reward:
                            print(p_copy.toString(), reward)
                            #with open(filename, "a") as text_file:
                            #    text_file.write(p_copy.toString()+str(reward)+"\n")
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

                    #"""
                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        if best_policy is not None:
            reward = self.evaluate(best_policy, 1000)
            print(best_policy.toString(), reward)
           # with open(filename, "a") as text_file:
           #     text_file.write("best: " +best_policy.toString() + str(reward) + "\n")

        best_rewards.append(best_reward)
        elapsed_times.append(time.time() - start)
        current_evaluation.append(number_evaluations)

        plt.clf()
        plt.step(elapsed_times, best_rewards, where='post')
        plt.title("CartPole-v1")
        plt.ylim(0, 500)
        plt.ylabel("Average Return")
        plt.xlabel("Elapsed Time (s)")
        plt.savefig(graph_name)
        plt.show()

        return best_policy, number_evaluations

    def synthesize_neuron(self, bound, operations, constant_values, observation_values, observations, actions, neuron, PiRL = False):

        start = time.time()

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, [], [])

        best_reward = -100 # distance instead of reward now...
        best_policy = None
        number_evaluations = 0

        best_rewards = []
        elapsed_times = []

        self.outputs = set()


        parameter_finder = ParameterFinderContinuous(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                print(closed_list)

                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        # optimize constant in decision rule, record distance
                        reward = parameter_finder.optimize_neuron(p_copy)
                    number_evaluations += 1

                    if reward > best_reward:

                        current_time = time.time() - start

                        print(p_copy.toString(), reward)

                        best_reward = reward
                        best_policy = p_copy

                        best_rewards.append(best_reward)
                        elapsed_times.append(time.time() - start)

                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        if best_policy is not None:
            print(best_policy.toString(), reward)

        return best_policy, number_evaluations

    def synthesize_neurons(self, bound, operations, constant_values, observation_values, observations, neurons, PiRL = False):

        # [[N1_0, N1_1, ..., N1_500], [N2_0, N2_1, ..., N2_500], ...]
        neuron_values = [trajs["N" + str(x)].to_numpy() for x in range(1, neurons+1, 1)]
        parameter_finder = [ParameterFinderContinuous(observations, neuron_values[i]) for i in range(neurons)]

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, [], [])

        best_reward = [-100] * neurons
        best_rewards = [[]] * neurons
        best_policy = [None] * neurons
        elapsed_times = [[]] * neurons
        number_evaluations = 0

        start = time.time()

        self.outputs = set()
        filenames = ["../PiRL-v2/Logs/ImitateNeurons/N_relu" + str(x) for x in range(1, neurons+1)]
        if PiRL:
            filenames = [x + "_PiRL.txt" for x in filenames]
        else:
            filenames = [x + ".txt" for x in filenames]

        for i in range(len(filenames)):
            with open(filenames[i], "w") as text_file:
                # Constants
                text_file.write("Constants: ")
                for c in constant_values:
                    text_file.write("%s, " % c)

                text_file.write("\nBest programs:\n")

        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                for i in range(neurons):
                    if p.name() == Ite.name():
                        p_copy = copy.deepcopy(p)
                        if PiRL:
                            # optimize constant in decision rule, record distance
                            reward = parameter_finder[i].optimize_neuron(p_copy)
                        number_evaluations += 1

                        if reward > best_reward[i]:
                            current_time = time.time() - start
                            print(p_copy.toString(), reward)
                            with open(filenames[i], "a") as text_file:
                                text_file.write(p_copy.toString()+str(reward)+"\n")
                            # Update best
                            best_reward[i] = reward
                            best_policy[i] = p_copy

                            best_rewards[i].append(best_reward[i])
                            elapsed_times[i].append(time.time() - start)

                        if number_evaluations % 1000 == 0:
                            print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        return best_policy, number_evaluations


if __name__ == '__main__':
    synthesizer = BottomUpSearch()

    OBSERVATIONS = np.load("./Oracle/2x4/1/Observations_50.npy")
    ACTIONS_c = np.load("./Oracle/2x4/1/Actions_50.npy")

    synthesizer = BottomUpSearch()
    neuron1_tree, num = synthesizer.synthesize_neuron(11, [Ite, Lt, AssignAction],
                                                      [0.0], [0, 1, 2, 3], OBSERVATIONS,
                                                      ACTIONS_c, 1, PiRL=True)
    exit()

    ## MountainCarContinuous-v0
    #observations = np.load("observations_con.npy")[:100]
    #actions = np.load("actions_con.npy")[:100]

    ## CartPole-v1
    #trajs = pd.read_csv("../Setup/trajectory.csv", nrows=5000)
    trajs = pd.read_csv("../../Setup/trajectory.csv")
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()
    actions = trajs['a'].to_numpy()

    prog_relus = []
    prog_relus.append(ReLU([1.1228,  0.0633, -3.4613, -2.2452], 0.1867))
    prog_relus.append(ReLU([-0.5183, -0.7270, -6.0250, -2.2768], 0.1656))

    p, num = synthesizer.synthesize(15, [Ite, Lt], [0.0], [0, 1, 2, 3], [0, 1], observations, actions,
                                    prog_relus, "_test_Nov", PiRL=False)

    exit()

    p, num = synthesizer.synthesize(15, [Ite, Lt], [-0.0462, -0.0068], [0, 1, 2, 3], [0, 1], observations, actions,
                                    [], "sept_test", PiRL=True)

