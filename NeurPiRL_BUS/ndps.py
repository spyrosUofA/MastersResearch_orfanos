import torch
import gym
from DSL import Ite, Lt, Observation, Num, AssignAction, Addition, Multiplication, ReLU
import numpy as np
import copy
from evaluation import DAgger
import pandas as pd
import pickle
import time

from stable_baselines3 import PPO
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


    def imitate_oracle(self, bound, eval_fn, operations, constant_values, observation_values, action_values,
                   boolean_programs, PiRL=False):

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, action_values, boolean_programs)

        best_score = 0.0
        best_policy = None
        number_evaluations = 0

        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    # Evaluate policy
                    if PiRL:
                        score = eval_fn.optimize(p_copy)
                    else:
                        score = eval_fn.evaluate(p_copy)
                    number_evaluations += 1

                    if score > best_score:
                        best_policy = p_copy
                        best_score = score
                        print("Score: ", best_score)
                        print(best_policy.toString())

                    #if number_evaluations % 1000 == 0:
                    #    print('AST Size: ', current_size, ' Evaluations: ', number_evaluations)

        return best_policy, number_evaluations



def algo_NDPS(oracle_path, roll_outs, seed=1, pomd='CartPole-v1'):

    # Task setup
    env = gym.make(pomd)
    np.random.seed(seed)
    env.seed(seed)

    operations = [Ite, Lt]
    constant_values = [0.0]
    observation_values = np.arange(env.observation_space.shape[0])
    action_values = np.arange(env.action_space.n)

    # load oracle model, histories, and relus
    model = PPO.load("./Oracle/" + oracle_path + '/model.zip')
    inputs = np.load("./Oracle/" + oracle_path + "/Observations.npy").tolist()
    actions = np.load("./Oracle/" + oracle_path + "/Actions.npy").tolist()
    relu_programs = pickle.load(open("./Oracle/" + oracle_path + "/ReLUs.pkl", "rb"))

    # Arguments for evaluation function
    oracle = {"oracle": model, "inputs": inputs, "actions": actions}
    synthesizer = BottomUpSearch()
    eval_fn = DAgger(oracle, nb_evaluations=25, seed=seed, env_name=pomd)

    # NDPS
    best_reward = 0
    for i in range(roll_outs):
        # Imitation Step
        next_program, nb_evals = synthesizer.imitate_oracle(11, eval_fn, operations, constant_values, observation_values, action_values,
                    relu_programs, True)

        # Evalaute program
        reward = eval_fn.collect_reward(next_program, 100)

        # Update program
        if reward > best_reward:
            best_reward = reward
            best_program = next_program

        # Update histories
        eval_fn.update_trajectory0(best_program)

        print("\nReward: ", best_reward, reward)
        print(best_program.toString())
        print("~~~~")

    return best_program

if __name__ == '__main__':
    algo_NDPS(oracle_path="2x4/1", roll_outs=5)

