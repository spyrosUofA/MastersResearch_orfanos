import gym
from DSL import Ite, Lt, Observation, Num, AssignAction
import numpy as np
import copy
from Optimization_so import ParameterFinder
import pandas as pd
import pickle

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
            p = AssignAction(i)
            self.insert(p)

        for i in range(len(boolean_programs)):
            self.insert(boolean_programs[i])
            # print(self.plist[1]['Lt'])
            
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
                ob, r_t, done, _ = env.step(action[0]) # changed
                
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

    def synthesize(self, bound, operations, constant_values, observation_values, action_values, observations, actions, boolean_programs, PiRL = False):

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, action_values, boolean_programs) # extra agrument here with set of programs

        best_reward = -1 #0
        best_policy = None
        number_evaluations = 0

        self.outputs = set()
        filename = "programs"
        if PiRL:
            filename+="_PiRL.txt"
        else:
            filename+=".txt"
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
                    #print(p_copy.toString())
                    #print("done")
                    #"""
                    reward = self.evaluate(p_copy, 10)
                    if reward > best_reward:
                        reward = self.evaluate(p_copy, 100)
                        if reward > best_reward:
                            print(p_copy.toString(), reward)
                            with open(filename, "a") as text_file:
                                text_file.write(p_copy.toString()+str(reward)+"\n")
                            best_reward = reward
                            best_policy = p_copy
                    #"""
                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        if best_policy is not None:
            reward = self.evaluate(best_policy, 1000)
            print(best_policy.toString(), reward)
            with open(filename, "a") as text_file:
                text_file.write("best: "+best_policy.toString() + str(reward) + "\n")
        return best_policy, number_evaluations

    def synthesize_neuron(self, bound, operations, constant_values, observation_values, action_values, observations, actions, PiRL = False):

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, action_values, []) # extra agrument here with set of programs

        best_reward = -100 # distance instead of reward now...
        best_policy = None
        number_evaluations = 0

        self.outputs = set()
        filename = "programs"
        if PiRL:
            filename+="_PiRL.txt"
        else:
            filename+=".txt"
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
                        # optimize constant in decision rule, record distance
                        reward = parameter_finder.optimize_neuron(p_copy)
                        #print(reward)
                        #print(p_copy.toString())
                    number_evaluations += 1

                    #reward = self.evaluate(p_copy, 10)
                    if reward > best_reward:
                        #reward = self.evaluate(p_copy, 100)
                        if reward > best_reward:
                            print(p_copy.toString(), reward)
                            with open(filename, "a") as text_file:
                                text_file.write(p_copy.toString()+str(reward)+"\n")
                            best_reward = reward
                            best_policy = p_copy
                    #"""
                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        if best_policy is not None:
            #reward = self.evaluate(best_policy, 1000)
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

    #"""
    print("Neuron 1")
    neuron1 = trajs['N1'].to_numpy()
    neuron1_tree, num = synthesizer.synthesize_neuron(10, [Ite, Lt], [0.1], [0, 1, 2, 3], [0, 0.5, 1, 2], observations, neuron1, PiRL=True)
    #pickle.dump(neuron1_tree, file=open("neuron1_tree.pickle", "wb"))

    exit()

    print("Neuron 2")
    neuron2 = trajs['N2'].to_numpy()
    neuron2_tree, num = synthesizer.synthesize_neuron(10, [Ite, Lt], [0.1], [0, 1, 2, 3], [0, 0.5, 1, 2], observations, neuron2, PiRL=True)
    pickle.dump(neuron2_tree, file=open("neuron2_tree.pickle", "wb"))
    #"""

    """
    a_min = min(actions)
    a_max = max(actions)
    print(a_min, a_max)
    some_values = np.unique([0.03,	-0.34,	-0.15,	0.10,	-0.04,	-0.15,	0.03,	0.22,	-0.34,	0.04,	0.00,	0.04,	0.22,	0.00,	-0.04,	0.05,	0.05,	0.10])
    """

    neuron1_tree = pickle.load(open("neuron1_tree.pickle", "rb")).getBooleans()
    neuron2_tree = pickle.load(open("neuron1_tree.pickle", "rb")).getBooleans()

    bool_programs = copy.deepcopy(neuron1_tree + neuron2_tree)
    #boolean_rules = re.findall(r'.if\((.*?)\)', p.toString())

    print(bool_programs)

    for i in range(len(bool_programs)):
        print(bool_programs[i].toString())
        bool_programs[i].size = 1

    actions = trajs['a'].to_numpy()

    p, num = synthesizer.synthesize(15, [Ite, Lt], [0.25], [0, 1, 2, 3], [0, 1], observations, actions, bool_programs, PiRL=False)


