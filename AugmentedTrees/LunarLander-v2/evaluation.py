from scipy import spatial
import numpy as np
from bayes_opt import BayesianOptimization
from DSL import *
from collections import deque
import gym

import copy
import torch

class Evaluate():

    def __init__(self, oracle, nb_evaluations, seed, env_name):
        self.seed = seed
        self.nb_evaluations = nb_evaluations
        self.games_played = 0

        # Make environment
        self.env = gym.make(env_name)
        self.env.seed(seed)

        # Load oracle parameters (optional)
        self.oracle = oracle.pop('oracle', None)
        self.capacity = oracle.pop('capacity', None)
        # Load oracle history
        if self.capacity is None:
            self.inputs = oracle.pop('inputs', None)
            self.actions = oracle.pop('actions', None)
        else:
            self.inputs = deque(oracle.pop('inputs', None), maxlen=int(self.capacity))
            self.actions = deque(oracle.pop('actions', None), maxlen=int(self.capacity))

    def get_games_played(self):
        return self.games_played

    def evaluate(self, p):
        pass

    def eval_triage(self, p):
        pass

    def update_trajectory0(self, p, nb_rollouts):
        pass

    def update_trajectory1(self, p, current_score):
        return current_score

    def collect_reward(self, p, nb_episodes, render=False):
        steps = 0
        averaged = 0.0
        for i in range(nb_episodes):
            ob = self.env.reset()
            reward = 0.0
            while True:
                if render:
                    self.env.render()
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = namespace['act']
                ob, r_t, done, _ = self.env.step(action)
                steps += 1
                reward += r_t

                if done:
                    self.env.close()
                    break
            averaged += reward
        self.games_played += nb_episodes

        return averaged / nb_episodes

# DISCRETE ACTIONS
    def find_distance(self, p):
        actions = get_action(self.inputs, p)
        actions_diff = spatial.distance.hamming(actions, np.array(self.actions))
        return  1.0 - actions_diff

# CONTINUOUS ACTIONS
#    def find_distance(self, p):
#        actions = get_action(self.inputs, p)
#        actions_diff = spatial.distance.euclidean(actions, np.array(self.actions))
#        return -actions_diff / float(len(self.actions))

    def find_distance_bo(self, **kwargs):
        # For bayesian optimization, pass dictionary of the Nums
        numNodes = kwargs # np.fromiter(kwargs.values(), dtype=float)
        self.tree.set_Num_value(numNodes)
        return self.find_distance(self.tree)

    def optimize(self, p):

        self.tree = p #copy.deepcopy(p)
        list_Nums_range, originals = p.get_Num_range()
        bayesOpt = BayesianOptimization(self.find_distance_bo, pbounds=list_Nums_range, verbose=0, random_state=self.seed)

        try:
            # Bayesian Optimization, then update tree with optimized values
            bayesOpt.maximize(init_points=40, n_iter=2, kappa=2.5)
            p.set_Num_value(bayesOpt.max['params'])

        except Exception as error:
            #print("No Nums to optimize, i.e., ", error)
            p.set_Num_value(originals)


class Environment(Evaluate):

    def __init__(self, oracle, nb_evaluations, seed, env_name):
        super(Environment, self).__init__(oracle, nb_evaluations, seed, env_name)
        self.worst_score = -1000.0

    def evaluate(self, p, render=False):
        return self.collect_reward(p, self.nb_evaluations, render)

    def eval_triage(self, p):
        pass

    def update_trajectory0(self, p, nb_rollouts=1):
        rew = 0.0
        for _ in range(nb_rollouts):
            ob = self.env.reset()
            while True:
                # Oracle's action recorded
                action_oracle, _ = self.oracle.predict(ob, deterministic=True)
                self.inputs.append(ob)
                self.actions.append(action_oracle)
                # PiRL's action executed in environemnt
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = namespace['act']
                # Interact with environment
                ob, r, done, _ = self.env.step([action])
                rew += r

                if done:
                    break

        self.games_played += nb_rollouts
        return rew / nb_rollouts


class Imitation(Evaluate):

    def __init__(self, oracle, nb_evaluations, seed, env_name):
        super(Imitation, self).__init__(oracle, nb_evaluations, seed, env_name)
        self.worst_score = 0.0

    def evaluate(self, p):
        return self.find_distance(p)


class DAgger(Evaluate):

    def __init__(self, oracle, nb_evaluations, seed, env_name):
        super(DAgger, self).__init__(oracle, nb_evaluations, seed, env_name)
        self.worst_score = 0.0

    def update_trajectory0(self, p, nb_rollouts=1):
        rew = 0.0
        for _ in range(nb_rollouts):  #range(self.nb_evaluations):
            ob = self.env.reset()
            while True:
                # Oracle's action recorded
                action_oracle, _ = self.oracle.predict(ob, deterministic=True)
                self.inputs.append(ob)
                self.actions.append(action_oracle)
                # PiRL's action executed in environemnt
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = namespace['act']
                # Interact with environment
                ob, r, done, _ = self.env.step(action)
                rew += r

                if done:
                    break

        self.games_played += nb_rollouts
        return rew / nb_rollouts

    def update_trajectory1(self, p, current_score):

        start_len = len(self.actions)
        correct = 0.0
        for _ in range(1):
            ob = self.env.reset()
            while True:
                # Oracle's action recorded
                #action_oracle = self.oracle.act(ob)[0]
                action_oracle, _ = self.oracle.predict(ob, deterministic=True)
                self.inputs.append(ob)
                self.actions.append(action_oracle)
                # PiRL's action to be executed in environment
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = namespace['act']
                # Compare actions, interact with environment
                correct += int(action_oracle == action)
                ob, _, done, _ = self.env.step(action)

                if done:
                    break

        self.games_played += 1 # self.nb_evaluations
        end_len = len(self.actions)
        return current_score * (start_len / end_len) + (correct / end_len)

    def evaluate(self, p):
        #self.update_trajectory(p)
        return self.find_distance(p)


def get_action(obs, p):
    actions = []
    for ob in obs:
        namespace = {'obs': ob, 'act': 0}
        p.interpret(namespace)
        actions.append(namespace['act'])
    return actions
