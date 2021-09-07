from scipy import spatial
import numpy as np
from bayes_opt import BayesianOptimization
import copy
from DSL import *


class Evaluate():

    def __init__(self, nb_evaluations, seed):
        self.nb_evaluations = nb_evaluations
        self.seed = seed
        self.worst_score = None

    def add_trajectory(self, inputs, actions):
        self.inputs = inputs
        self.actions = actions

    def get_Num_range(self):
        dict_ranges = {}
        originals = []
        i = 1
        # BFS
        q = []
        q.append(self.tree)
        while len(q) > 0:
            node = q.pop(0)
            #print(node)
            if type(node) is Num:
                name = "Num" + str(i)
                i += 1
                originals.append(node.children[0])
                interval = create_interval(node.children[0], 0.1)
                dict_ranges[name] = copy.deepcopy(interval)
                # print(type(interval))
            elif type(node) is Ite:
                q.append(node.children[0])
                q.append(node.children[1])
                q.append(node.children[2])
            elif type(node) is Lt:
                q.append(node.children[0])
                q.append(node.children[1])
            # elif type(node) is AssignAction:
            #    q.append(node.value)
            elif type(node) is Addition:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is StartSymbol:
                q.append(node.children[0])
        return dict_ranges, originals

    # traverse tree, whenever we find Num node we set the value of the node according to [name].
    def set_Num_value(self, values):
        # BFS
        q = []
        i = 1
        q.append(self.tree)
        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Num:
                name = "Num" + str(i)
                i += 1
                if type(values) is not list:
                    node.children[0] = values[name]
                else:
                    node.children[0] = values.pop(0)
            elif type(node) is Ite:
                q.append(node.children[0])
                q.append(node.children[1])
                q.append(node.children[2])
            elif type(node) is Lt:
                q.append(node.children[0])
                q.append(node.children[1])
            # elif type(node) is AssignAction:
            #    q.append(node.value)
            elif type(node) is Addition:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is StartSymbol:
                q.append(node.children[0])
        return

    def evaluate(self, p):
        pass

    def eval_triage(self, p):
        pass

    def find_distance(self, **kwargs):
        numNodes = np.fromiter(kwargs.values(), dtype=float)
        self.set_Num_value(numNodes.tolist())

        return self.evaluate(self.tree)[0]

    def optimize(self, tree):
        self.tree = tree

        # list of Nums in the AST to optimize over
        list_Nums_range, originals = self.get_Num_range()
        bayesOpt = BayesianOptimization(self.find_distance, pbounds=list_Nums_range, verbose=0)

        try:
            # Bayesian Optimization
            bayesOpt.maximize(init_points=40, n_iter=10, kappa=2.5)
            # Update tree with optimized Nums
            self.set_Num_value(bayesOpt.max['params'])
            return bayesOpt.max['target']

        except Exception as error:
            #print("No Nums to optimize, i.e., ", error)
            self.set_Num_value(originals)
            return originals


class Environment(Evaluate):

    def __init__(self, nb_evaluations, seed, env_name='LunarLander-v2'):
        self.nb_evaluations = nb_evaluations
        self.worst_score = -500.0
        import gym
        self.env = gym.make(env_name)
        self.env.seed(seed)

    def evaluate(self, p):
        steps = 0
        averaged = 0.0

        for _ in range(self.nb_evaluations):
            ob = self.env.reset()
            reward = 0.0
            while True:
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                ob, r_t, done, _ = self.env.step(action[0])
                steps += 1
                reward += r_t

                if done:
                    break
            averaged += reward

        return averaged / self.nb_evaluations, self.nb_evaluations

    def eval_render(self, p):
        steps = 0
        averaged = 0.0

        for _ in range(self.nb_evaluations):
            ob = self.env.reset()
            reward = 0.0
            while True:
                self.env.render()
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                ob, r_t, done, _ = self.env.step(action[0])
                steps += 1
                reward += r_t

                if done:
                    self.env.close()
                    print(reward)
                    break
            averaged += reward

        return averaged / self.nb_evaluations, self.nb_evaluations

    def eval_triage(self, p):
        steps = 0
        averaged = 0

        for _ in range(self.nb_evaluations):
            ob = self.env.reset()
            reward = 0
            while True:
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                ob, r_t, done, _ = self.env.step(action[0])
                steps += 1
                reward += r_t

                if done: break
            averaged += reward

        return averaged / self.nb_evaluations, self.nb_evaluations


class Imitation(Evaluate):

    def __init__(self, nb_evaluations, seed):
        self.nb_evaluations = nb_evaluations
        self.worst_score = 0.0
        # For Neural Policy
        self.inputs = None
        self.actions = None

    def evaluate(self, p):
        # self.actions --> produced by Oracle.
        # actions --> produced by AST.
        actions = get_action(self.inputs, p)
        actions_diff = spatial.distance.hamming(actions, np.array(self.actions))

        return 1 - actions_diff, 0


def get_action(obs, p):
    actions = []
    for ob in obs:
        namespace = {'obs': ob, 'act': 0}
        p.interpret(namespace)
        actions.append(namespace['act'])
    return actions


def create_interval(value, delta):
    interval = (value - delta, value + delta)
    return interval