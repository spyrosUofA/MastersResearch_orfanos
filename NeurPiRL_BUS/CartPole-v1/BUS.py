import gym
from DSL import Ite, Lt, Observation, Num, AssignAction, ReLU
import numpy as np
import copy
from OptimizationDiscrete import ParameterFinderDiscrete
import pandas as pd
import pickle
import time

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

    def init_plist(self, constant_values, observation_values, action_values, relu_programs):
        for i in observation_values:
            p = Observation(i)
            self.insert(p)

        for i in constant_values:
            p = Num(i)
            self.insert(p)

        for i in action_values:
            p = AssignAction(Num(i))
            self.insert(p)

        for i in range(len(relu_programs)):
            p = ReLU(relu_programs[i][0], relu_programs[i][1])
            self.insert(p)

    def get_programs(self, size):

        if size in self.plist:
            return self.plist[size]
        return {}


class BottomUpSearch():

    def evaluate(self, p, episode_count, env):
        steps = 0
        averaged = 0

        for _ in range(episode_count):
            ob = env.reset()
            reward = 0
            while True:
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                # print(action)
                ob, r_t, done, _ = env.step(action[0])  # changed

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
                   relu_programs, name, PiRL=False):

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, action_values,
                         relu_programs)  # extra agrument here with set o fprograms

        best_reward = 0
        best_policy = None
        number_evaluations = 0
        score = None

        self.outputs = set()

        env = gym.make('CartPole-v1')
        env.seed(0)

        # configure files
        p_file = "./programs/" + name
        l_file = "./logs/" + name
        if PiRL:
            p_file += "_PiRL.txt"
            l_file += "_PiRL.txt"
        else:
            p_file += ".txt"
            l_file += ".txt"

        with open(p_file, "w") as text_file:
            text_file.write("Best programs:\n")

        t0 = time.time()

        parameter_finder = ParameterFinderDiscrete(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        parameter_finder.optimize(p_copy)
                    number_evaluations += 1
                    reward = self.evaluate(p_copy, 25, env)

                    if reward > best_reward:
                        reward = self.evaluate(p_copy, 100, env)
                        if reward > best_reward:
                            best_reward = reward
                            best_policy = p_copy

                            t1 = time.time() - t0
                            print(p_copy.toString(), reward)

                            # Log
                            with open(l_file, 'a') as results_file:
                                results_file.write(("{:f}, {:d}, {:f}\n".format(t1, number_evaluations, best_reward)))
                            # Policy
                            with open(p_file, 'a') as results_file:
                                results_file.write(p_copy.toString() + str(reward) + "\n")

                            if reward > 499.9:
                                break

                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        return best_policy, number_evaluations


if __name__ == '__main__':

    for oracle in range(1, 2):

        # WITH SMALL AUGMENTED DSL
        path_to = "2x4/" + str(oracle)
        BOUND = 11
        OPERATIONS = [Ite, Lt]
        CONSTANTS = [0.0]
        OBS_VALUES = []
        ACT_VALUES = [0, 1]
        RELU_PROG = pickle.load(open("./Oracle/" + path_to + "/ReLUs.pkl", "rb"))
        OBSERVATIONS = np.load("./Oracle/" + path_to + "/Observations_50.npy")
        ACTIONS = np.load("./Oracle/" + path_to + "/Actions_50.npy")
        synthesizer = BottomUpSearch()


        #p, num = synthesizer.synthesize(BOUND, OPERATIONS, CONSTANTS, OBS_VALUES, ACT_VALUES, OBSERVATIONS, ACTIONS,
        #                                RELU_PROG, path_to, PiRL=True)

        # WITH BIG AUGMENTED DSL
        path_to = "64x64/" + str(oracle)
        BOUND = 11
        OPERATIONS = [Ite, Lt]
        CONSTANTS = [0.0]
        OBS_VALUES = []
        ACT_VALUES = [0, 1]
        RELU_PROG = pickle.load(open("./Oracle/" + path_to + "/ReLUs.pkl", "rb"))
        OBSERVATIONS = np.load("./Oracle/" + path_to + "/Observations_50.npy")
        ACTIONS = np.load("./Oracle/" + path_to + "/Actions_50.npy")
        synthesizer = BottomUpSearch()

        p = Ite(
            Lt(Observation(3), Num(0.22)),
            Ite(
                Lt(Num(0.01), Observation(2)),
                AssignAction(Num(1)),
                AssignAction(Num(0))),
            AssignAction(Num(1)))
        a1 = get_action(OBSERVATIONS, p)

        env = gym.make("CartPole-v1")
        print(np.sum(a1 == ACTIONS) / len(a1))
        print(synthesizer.evaluate(p, 500, env))


        exit()

        #p, num = synthesizer.synthesize(BOUND, OPERATIONS, CONSTANTS, OBS_VALUES, ACT_VALUES, OBSERVATIONS, ACTIONS,
        #                                RELU_PROG, path_to, PiRL=True)

        # WITHOUT AUGMENTED DSL
        trajs = pd.read_csv("./trajectory.csv")
        observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()
        actions = trajs['a'].to_numpy()

        p = Ite(
            Lt(Observation(3), Num(0.22)),
                Ite(
                    Lt(Num(0.01), Observation(2)),
                    AssignAction(Num(1)),
                    AssignAction(Num(0))),
                AssignAction(Num(1)))
        a1 = get_action(observations, p)

        env = gym.make("CartPole-v1")
        print(np.sum(a1 == actions) / len(a1))
        print(synthesizer.evaluate(p, 500, env))
        exit()

        path_to = "BaseDSL/000" + str(oracle)
        BOUND = 11
        OPERATIONS = [Ite, Lt]
        CONSTANTS = [0.0] #1, 0.22]
        OBS_VALUES = [0, 1, 2, 3]
        ACT_VALUES = [0, 1]
        RELU_PROG = []
        OBSERVATIONS = observations
        ACTIONS = actions
        #OBSERVATIONS = np.load("./Oracle/64x64/" + path_to + "/Observations_50.npy")
        #ACTIONS = np.load("./Oracle/64x64/" + path_to + "/Actions_50.npy")
        synthesizer = BottomUpSearch()

        p, num = synthesizer.synthesize(BOUND, OPERATIONS, CONSTANTS, OBS_VALUES, ACT_VALUES, OBSERVATIONS, ACTIONS,
                                        RELU_PROG, path_to, PiRL=True)
