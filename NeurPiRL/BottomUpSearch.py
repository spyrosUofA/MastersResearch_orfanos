import gym
from DSL import Ite, Lt, Observation, Num, AssignAction, Addition, Multiplication, ReLU
import numpy as np
import copy
from Optimization import ParameterFinderContinuous, ParameterFinderDiscrete


import pandas as pd
import pickle
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib

import warnings
warnings.filterwarnings('error')

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

        with open(filename, "w") as text_file:
            text_file.write("Best programs:\n")
        parameter_finder = ParameterFinderDiscrete(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                #print(p.name)
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        #print("test")
                        #print(p_copy.toString())
                        reward = parameter_finder.optimize(p_copy)
                        #print(reward)
                        #print(p_copy.toString())
                    number_evaluations += 1
                    reward = self.evaluate(p_copy, 10)
                    if reward > best_reward:
                        reward = self.evaluate(p_copy, 100)
                        if reward > best_reward:
                            print(p_copy.toString(), reward)
                            with open(filename, "a") as text_file:
                                text_file.write(p_copy.toString()+str(reward)+"\n")
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
            with open(filename, "a") as text_file:
                text_file.write("best: "+best_policy.toString() + str(reward) + "\n")

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
        filename = "../PiRL-v2/Logs/ImitateNeurons/N" + str(neuron)

        if PiRL:
            filename += "_PiRL.txt"
        else:
            filename += ".txt"
        with open(filename, "w") as text_file:
            text_file.write("Best programs:\n")
        parameter_finder = ParameterFinderContinuous(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        # optimize constant in decision rule, record distance
                        reward = parameter_finder.optimize_neuron(p_copy)
                    number_evaluations += 1

                    if reward > best_reward:

                        current_time = time.time() - start

                        print(p_copy.toString(), reward)
                        with open(filename, "a") as text_file:
                            text_file.write(p_copy.toString()+str(reward)+str(current_time)+"\n")
                        best_reward = reward
                        best_policy = p_copy

                        best_rewards.append(best_reward)
                        elapsed_times.append(time.time() - start)

                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        if best_policy is not None:
            print(best_policy.toString(), reward)
            with open(filename, "a") as text_file:
                text_file.write("best: "+best_policy.toString() + str(reward) + "\n")
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

    ## MountainCarContinuous-v0
    #observations = np.load("observations_con.npy")[:100]
    #actions = np.load("actions_con.npy")[:100]

    ## CartPole-v1
    trajs = pd.read_csv("../Setup/trajectory.csv", nrows=5000)
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()

    NEURON_CONSTANTS = [-0.25, 0.0, 0.25, 0.5, 0.8, 2]
    NUM_CONSTANTS = [0.25, 0.0, -0.25]
    NEURONS = 2

    prog_relus = []
    prog_relus.append(ReLU([1.1228,  0.0633, -3.4613, -2.2452], 0.1867))
    prog_relus.append(ReLU([-0.5183, -0.7270, -6.0250, -2.2768], 0.1656))

    obs0 = [-0.04456399, 0.04653909, 0.01326909, -0.02099827]
    #print(prog_relus[0].interpret(obs0).toString())
    #print(prog_relus[1].interpret(obs0).toString())

    namespace = {'obs': obs0, 'act': 0}
    print(namespace['obs'])
    print(type(namespace['obs']))
    print(prog_relus[0].interpret(namespace))

    actions = trajs['a'].to_numpy()
    p, num = synthesizer.synthesize(11, [Ite, Lt], NUM_CONSTANTS, [0, 1, 2, 3], [0, 1], observations, actions,
                                    prog_relus, "_relu_test2a", PiRL=True)
    #Found array with 0 feature(s) (shape=(1, 0)) while a minimum of 1 is required.
    exit()

    #neuron_values = [trajs["N" + str(x)].to_numpy() for x in lala]
    neuron1_tree, num = synthesizer.synthesize_neurons(10, [Ite, Lt, AssignAction, Addition], NEURON_CONSTANTS,
                                                      [0, 1, 2, 3], observations, NEURONS, PiRL=True)
    exit()

    # Imitate Neurons
    if False:
        starting_time = time.time()

        print("Neuron 1")
        neuron1 = trajs['N1'].to_numpy()
        neuron1_tree, num = synthesizer.synthesize_neuron(10, [Ite, Lt, AssignAction, Addition], NEURON_CONSTANTS, [0, 1, 2, 3], observations, neuron1, 1, PiRL=True)
        pickle.dump(neuron1_tree, file=open("neuron1_tree_test.pickle", "wb"))
        #(if(obs[3] < 0.1068607496172635) then act = 0.4 else act = 2.0) -0.019605807716307922
        #neuron1_tree = Ite(Lt(Observation(1), Num(-0.33849702456845226)), AssignAction(Num(1.985333726874753)),
        #                   Ite(Lt(Observation(1), Num(0.0030069121715175368)), AssignAction(Num(0.8942757185406707)),
        #                       AssignAction(Num(0.0030069121715175368))))
        print("N1 time:")
        print(time.time() - starting_time)

        print("Neuron 2")
        neuron2 = trajs['N2'].to_numpy()
        neuron2_tree, num = synthesizer.synthesize_neuron(10, [Ite, Lt, AssignAction, Addition], NEURON_CONSTANTS, [0, 1, 2, 3],  observations, neuron2, 2, PiRL=True)
        pickle.dump(neuron2_tree, file=open("neuron2_tree.pickle", "wb"))
        end_time = time.time() - starting_time

        print("total time:")
        print(time.time() - starting_time)


    if False:
        neuron1_tree = pickle.load(open("neuron1_tree.pickle", "rb")).getBooleans()
        neuron2_tree = pickle.load(open("neuron2_tree.pickle", "rb")).getBooleans()
        bool_programs = copy.deepcopy(neuron1_tree + neuron2_tree)
        #boolean_rules = re.findall(r'.if\((.*?)\)', p.toString())

        for i in range(len(bool_programs)):
            print(bool_programs[i].toString())
            bool_programs[i].size = 1

    if False:
        bool_programs = []
        bool_programs.append(Lt(Num(0.02), Observation(1)))
        bool_programs.append(Lt(Observation(1), Num(0.02)))
        bool_programs.append(Lt(Num(-0.34), Observation(1)))
        bool_programs.append(Lt(Num(0.03), Observation(2)))
        bool_programs.append(Lt(Observation(1), Num(-0.17)))
        bool_programs.append(Lt(Observation(2), Num(0.0)))
        bool_programs.append(Lt(Num(-0.17), Observation(1)))
        bool_programs.append(Lt(Observation(3), Num(-0.01)))
        bool_programs.append(Lt(Num(-0.01), Observation(3)))

        for i in range(len(bool_programs)):
            print(bool_programs[i].toString())
            bool_programs[i].size = 1

    actions = trajs['a'].to_numpy()
    #p, num = synthesizer.synthesize(15, [Ite, Lt], [0.25, 0.0, -0.25], [0, 1, 2, 3], [0, 1], observations, actions,
    #                                [], "_direct_15", PiRL=True)

    p, num = synthesizer.synthesize(15, [Ite, Lt], [-0.0462, -0.0068], [0, 1, 2, 3], [0, 1], observations, actions,
                                    [], "_direct_test", PiRL=False)

