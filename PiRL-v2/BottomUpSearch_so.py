import gym
from DSL import Ite, Lt, Observation, Num, AssignAction, Addition, Multiplication
import numpy as np
import copy
from Optimization_so import ParameterFinder
from Optimization_Continuous import ParameterFinder as ParameterFinderCon

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
        filename = "../PiRL-v2/Logs/ImitateNeurons/N" + str(neuron)

        if PiRL:
            filename += "_PiRL.txt"
        else:
            filename += ".txt"
        with open(filename, "w") as text_file:
            text_file.write("Best programs:\n")
        parameter_finder = ParameterFinderCon(observations, actions)
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


        # [[N1_0, N1_1, ..., N1_500], [N2_0, N2_1, ..., N2_500]]
        neuron_values = [trajs["N" + str(x)].to_numpy() for x in range(1, neurons+1, 1)]
        parameter_finder = [ParameterFinderCon(observations, neuron_values[i]) for i in range(neurons)]
        #parameter_finder = ParameterFinderCon(observations, actions)

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, [], [])

        best_reward = [-100] * neurons
        best_rewards = [None] * neurons
        best_policy = [None] * neurons
        elapsed_times = [[]] * neurons

        number_evaluations = 0

        start = time.time()

        self.outputs = set()
        filenames = ["../PiRL-v2/Logs/ImitateNeurons/N" + str(x) for x in range(1,neurons+1)]
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
                                text_file.write(p_copy.toString()+str(reward) + " " + str(current_time)+"\n")
                            best_reward[i] = reward
                            best_policy[i] = p_copy

                            #best_rewards[i].append(best_reward[i])
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


    #neuron_values = [trajs["N" + str(x)].to_numpy() for x in lala]
    neuron1_tree, num = synthesizer.synthesize_neurons(10, [Ite, Lt, AssignAction, Addition], NEURON_CONSTANTS,
                                                      [0, 1, 2, 3], observations, 2, PiRL=True)
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

"""
/usr/local/bin/python3.8 /Users/spyros/Documents/GitHub/MastersResearch_orfanos/PiRL-v2/BottomUpSearch_so.py
Neuron 1
(if(obs[0] < -0.15) then act = -0.15 else act = 0.1) -0.016261441684243705
(if(obs[0] < -0.2848038791543404) then act = -0.2848038791543404 else act = 0.35) -0.01581572204225666
(if(obs[0] < -0.3159866128750433) then act = -0.3159866128750433 else act = 0.4) -0.015810816046955053
(if(obs[0] < -0.35) then act = 0.1 else act = 0.35) -0.01456178805190415
(if(obs[0] < -0.35) then act = 0.1 else act = 0.4) -0.014540148042988569
(if(obs[0] < -0.15) then act = 0.35 else act = 0.1) -0.014122376515751182
(if(obs[0] < -0.35) then act = 0.35 else act = 0.41496964845024653) -0.013897202888006874
(if(obs[0] < -0.15) then act = 0.6 else act = 0.1) -0.013277708192829491
(if(obs[0] < -0.15) then act = 0.6 else act = 0.3091336680776624) -0.013048882771940593
(if(obs[0] < -0.15) then act = 0.9 else act = 0.1) -0.012520181889073355
(if(obs[0] < -0.15) then act = 0.9 else act = 0.30472619826096425) -0.012276899590281777
(if(obs[0] < -0.22525386074780954) then act = 1.9 else act = 0.1) -0.012023450561694647
(if(obs[0] < -0.2794081109632721) then act = 2.1 else act = 0.35) -0.011399224292172425
(if(obs[0] < -0.3118902682972835) then act = 2.0961685508449293 else act = 0.4244691352047755) -0.011337584092550604
(if(obs[1] < -0.17040992030135468) then act = 1.9268728577534344 else act = 0.018563623257113318) -0.010734631249294885
(if(obs[1] < -0.22221129385479005) then act = 2.0671187239926114 else act = 0.2906574821882436) -0.010171856205336172
AST Size:  6  Evaluations:  1000
AST Size:  8  Evaluations:  2000
(if(obs[0] < (obs[3] + -0.2125124246162683)) then act = 1.9214177175881513 else act = -0.2125124246162683) -0.009113211166526665
(if(obs[0] < (obs[3] + -0.30330897500559073)) then act = 2.0161984447622 else act = 0.1) -0.007931898572698706
(if(obs[0] < (obs[3] + -0.31549302534381163)) then act = 2.0103355601327504 else act = 0.2129164748107089) -0.007898797725028986
AST Size:  8  Evaluations:  3000
AST Size:  8  Evaluations:  4000
AST Size:  8  Evaluations:  5000
AST Size:  8  Evaluations:  6000
AST Size:  8  Evaluations:  7000
AST Size:  8  Evaluations:  8000
AST Size:  8  Evaluations:  9000
(if(obs[0] < (obs[3] + -0.31549302534381163)) then act = 2.0103355601327504 else act = 0.2129164748107089) -0.013664134537155107
N1 time:
7610.649210929871
Neuron 2
(if(obs[0] < -0.15) then act = -0.15 else act = 0.1) -0.00337968603544347
(if(obs[0] < -0.15) then act = 0.0030110101865351915 else act = 0.15) -0.003213014422960186
(if(obs[0] < 0.05825294525940214) then act = 0.05825294525940214 else act = 0.17720715830325198) -0.003071339845970812
(if(obs[0] < 0.18954172136191133) then act = 0.013429484518850773 else act = 0.18954172136191133) -0.003017690664556316
(if(0.21176443768462966 < obs[0]) then act = 0.21176443768462966 else act = 0.03890872835016703) -0.0030137417413876287
AST Size:  6  Evaluations:  1000
(if(obs[0] < (obs[1] + 0.35)) then act = 0.03600886722801245 else act = 0.35) -0.00292439328114489
(if(obs[0] < (obs[1] + 0.4)) then act = 0.04707498755130101 else act = 0.29991603726695903) -0.0029239437579932
(if(obs[0] < (obs[1] + 0.4)) then act = 0.040791885497634225 else act = 0.4) -0.0029218547846291926
AST Size:  8  Evaluations:  2000
AST Size:  8  Evaluations:  3000
AST Size:  8  Evaluations:  4000
AST Size:  8  Evaluations:  5000
AST Size:  8  Evaluations:  6000
AST Size:  8  Evaluations:  7000
(if((obs[1] + 0.4) < obs[0]) then act = 0.4 else act = 0.05036236797054431) -0.002920128547375977
AST Size:  8  Evaluations:  8000
AST Size:  8  Evaluations:  9000
(if((obs[1] + 0.4) < obs[0]) then act = 0.4 else act = 0.05036236797054431) -0.009217763726723956
total time:
15362.684953927994

Process finished with exit code 0
"""