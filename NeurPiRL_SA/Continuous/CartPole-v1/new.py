import gym
from DSL import Ite, Lt, Observation, Num, AssignAction, ReLU
import numpy as np
import copy
from OptimizationDiscrete import ParameterFinderDiscrete
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
        plist.init_plist(constant_values, observation_values, action_values, relu_programs)  # extra agrument here with set o fprograms

        best_reward = 0
        best_policy = None
        number_evaluations = 0
        score = None

        self.outputs = set()

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

        parameter_finder = ParameterFinderDiscrete(observations, actions)
        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    if PiRL:
                        score = parameter_finder.optimize(p_copy)

                    number_evaluations += 1
                    reward = self.evaluate(p_copy, 10)
                    if reward > best_reward:
                        reward = self.evaluate(p_copy, 100)
                        if reward > best_reward:
                            print(score)
                            print(p_copy.toString(), reward)
                            with open(filename, "a") as text_file:
                                text_file.write(p_copy.toString() + str(reward) + "\n")
                            best_reward = reward
                            best_policy = p_copy

                            if reward > 499.9:
                                break
                    # """
                    if number_evaluations % 1000 == 0:
                        print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        if best_policy is not None:
            reward = self.evaluate(best_policy, 1000)
            print(best_policy.toString(), reward)
            with open(filename, "a") as text_file:
                text_file.write("best: "+best_policy.toString() + str(reward) + "\n")
        return best_policy, number_evaluations



if __name__ == '__main__':
    synthesizer = BottomUpSearch()

    ## MountainCarContinuous-v0
    # observations = np.load("observations_con.npy")[:100]
    # actions = np.load("actions_con.npy")[:100]

    ## CartPole-v1
    trajs = pd.read_csv("../Setup/trajectory.csv")
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()
    actions = trajs['a'].to_numpy()

    # WITH RELUS
    BOUND = 20
    OPERATIONS = [Ite, Lt]
    CONSTANTS = [0.0]
    OBS_VALUES = []
    ACT_VALUES = [0, 1]
    RELU_PROG = pickle.load(open("./CartPole-v1/Oracle/2x4/15/ReLUs.pkl", "rb"))
    OBSERVATIONS = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()
    ACTIONS = trajs['a'].to_numpy()

    p, num = synthesizer.synthesize(BOUND, OPERATIONS, CONSTANTS, OBS_VALUES, ACT_VALUES, OBSERVATIONS, ACTIONS,
                                    RELU_PROG, PiRL=True)
    exit()
    # WITHOUT
    OPERATIONS = [Ite, Lt]
    CONSTANTS = [0.0]
    OBS_VALUES = [0, 1, 2, 3]
    ACT_VALUES = [0, 1]
    RELU_PROGS = pickle.load(open("Oracle/64x64/3/ReLUs.pkl", "rb"))
    OBSERVATIONS = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]']].to_numpy()
    ACTIONS = trajs['a'].to_numpy()
    p, num = synthesizer.synthesize(BOUND, OPERATIONS, CONSTANTS, OBS_VALUES, ACT_VALUES, RELU_PROGS, OBSERVATIONS,
                                    ACTIONS, PiRL=True)

    # Passed:
    #prog_relus = pickle.load(open("./Oracle/2x4/2/ReLUs.pkl", "rb"))
    #prog_relus = pickle.load(open("./Oracle/2x4/3/ReLUs.pkl", "rb"))

    # Current
    prog_relus = pickle.load(open("Oracle/64x64/3/ReLUs.pkl", "rb"))

#    some_values = np.unique(
#        [0.03, -0.34, -0.15, 0.10, -0.04, -0.15, 0.03, 0.22, -0.34, 0.04, 0.00, 0.04, 0.22, 0.00, -0.04, 0.05, 0.05,
#         0.10])
#    c = [0.01, 0.22] # (if(obs[2] < 0.01) then (if(0.22 < obs[3]) then act = 1 else act = 0) else act = 1) 488.232


    # synthesize(self, bound, operations, constant_values, observation_values, action_values, observations, actions, PiRL = False):
    p, num = synthesizer.synthesize(bound=20, operations=[Ite, Lt], constant_values=[0.0],
                                    observation_values=[0, 1, 2, 3], action_values=[0, 1], relu_programs=prog_relus,
                                    observations=observations, actions=actions, PiRL=False)

    print(p.toString())


    # p, num = synthesizer.synthesize(15, [Ite, Lt], some_values, [0, 1, 2, 3], [0, 1], observations, actions, PiRL=False)
    # (if(0.05689054185958309 < max(0, [-0.5183, -0.727, -6.025, -2.2768] * obs + 0.1656)) then act = 0 else act = 1) 500.0