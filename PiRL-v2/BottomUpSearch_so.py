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
        #actions.append(namespace['act'].value)  # EDITED...
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
        parameter_finder = ParameterFinder(observations, actions)
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
        filename = "neuron_" + str(neuron)

        if PiRL:
            filename += "_PiRL.txt"
        else:
            filename += ".txt"
        with open(filename, "w") as text_file:
            text_file.write("Best programs:\n")
        parameter_finder = ParameterFinder(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        #print(p_copy.toString())
                        # optimize constant in decision rule, record distance
                        reward = parameter_finder.optimize_neuron(p_copy)
                        #print(p_copy.toString())
                        #print(reward)
                    number_evaluations += 1

                    if reward > best_reward:
                        print(p_copy.toString(), reward)
                        with open(filename, "a") as text_file:
                            text_file.write(p_copy.toString()+str(reward)+"\n")
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


if __name__ == '__main__':
    synthesizer = BottomUpSearch()

    ## MountainCarContinuous-v0
    #observations = np.load("observations_con.npy")[:100]
    #actions = np.load("actions_con.npy")[:100]

    ## CartPole-v1
    trajs = pd.read_csv("../Setup/trajectory.csv")
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()

    # Imitate Neurons
    if False:
        print("Neuron 1")
        neuron1 = trajs['N1'].to_numpy()
        neuron1_tree, num = synthesizer.synthesize_neuron(10, [Ite, Lt, AssignAction, Addition], [-0.25, 0.0, 0.25, 0.5, 0.8, 2], [0, 1, 2, 3], observations, neuron1, 1, PiRL=True)
        pickle.dump(neuron1_tree, file=open("neuron1_tree.pickle", "wb"))
        #(if(obs[3] < 0.1068607496172635) then act = 0.4 else act = 2.0) -0.019605807716307922
        #neuron1_tree = Ite(Lt(Observation(1), Num(-0.33849702456845226)), AssignAction(Num(1.985333726874753)),
        #                   Ite(Lt(Observation(1), Num(0.0030069121715175368)), AssignAction(Num(0.8942757185406707)),
        #                       AssignAction(Num(0.0030069121715175368))))

        print("Neuron 2")
        neuron2 = trajs['N2'].to_numpy()
        neuron2_tree, num = synthesizer.synthesize_neuron(10, [Ite, Lt, AssignAction, Addition],  [-0.25, 0.0, 0.25, 0.5, 0.8, 2], [0, 1, 2, 3],  observations, neuron2, 2, PiRL=True)
        pickle.dump(neuron2_tree, file=open("neuron2_tree.pickle", "wb"))

    #""" Imitate Neural Policy
    neuron1_tree = pickle.load(open("neuron1_tree.pickle", "rb")).getBooleans()
    neuron2_tree = pickle.load(open("neuron2_tree.pickle", "rb")).getBooleans()
    bool_programs = copy.deepcopy(neuron1_tree + neuron2_tree)
    #boolean_rules = re.findall(r'.if\((.*?)\)', p.toString())

    if True:
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
    p, num = synthesizer.synthesize(15, [Ite, Lt], [0.25, 0.0, -0.25], [0, 1, 2, 3], [0, 1], observations, actions,
                                    [], "_direct", PiRL=True)

    exit()