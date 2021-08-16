from scipy import spatial
import numpy as np
from bayes_opt import BayesianOptimization
import copy
from DSL import *

class Evaluate():

    def __init__(self, nb_evaluations):
        self.nb_evaluations = nb_evaluations
        # For Neural Policy
        self.inputs = None
        self.actions = None

    def add_trajectory(self, inputs, actions):
        self.inputs = inputs
        self.actions = actions

    def evaluate(self, p):
        steps = 0
        averaged = 0.0

        import gym
        env = gym.make('LunarLander-v2')

        for _ in range(self.nb_evaluations):
            ob = env.reset()
            reward = 0.0
            while True:
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                ob, r_t, done, _ = env.step(action[0])
                steps += 1
                reward += r_t

                if done:
                    break
            averaged += reward

        return averaged / self.nb_evaluations, self.nb_evaluations

    def eval_render(self, p):
        steps = 0
        averaged = 0.0

        import gym
        env = gym.make('LunarLander-v2')

        for _ in range(self.nb_evaluations):
            ob = env.reset()
            reward = 0.0
            while True:
                env.render()
                namespace = {'obs': ob, 'act': 0}
                p.interpret(namespace)
                action = [namespace['act']]
                ob, r_t, done, _ = env.step(action[0])
                steps += 1
                reward += r_t

                if done:
                    env.close()
                    print(reward)
                    break
            averaged += reward

        return averaged / self.nb_evaluations

    def eval_triage(self, p):
        steps = 0
        averaged = 0

        import gym
        env = gym.make('LunarLander-v2')

        for _ in range(self.nb_evaluations):
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

        return averaged / self.nb_evaluations, self.nb_evaluations

    def find_distance(self, p):

        # self.actions --> produced by Oracle.
        # actions --> produced by AST.
        actions = get_action(self.inputs, p)
        actions_diff = spatial.distance.euclidean(actions, np.array(self.actions))

        diff_total = -(actions_diff) / float(len(self.actions))

        return diff_total, 0

class Imitation():

    def __init__(self, nb_evaluations):
        self.nb_evaluations = nb_evaluations
        # For Neural Policy
        self.inputs = None
        self.actions = None

    def get_Num_range(tree):
        dict_ranges = {}
        originals = []
        i = 1
        # BFS
        q = []
        q.append(tree)
        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Num:
                name = "Num" + str(i)
                i += 1
                originals.append(node.value)
                interval = create_interval(node.value, 0.1)
                dict_ranges[name] = copy.deepcopy(interval)
                # print(type(interval))
            elif type(node) is Ite:
                q.append(node.condition)
                q.append(node.true_case)
                q.append(node.false_case)
            elif type(node) is Lt:
                q.append(node.left)
                q.append(node.right)
            # elif type(node) is AssignAction:
            #    q.append(node.value)
            elif type(node) is Addition:
                q.append(node.left)
                q.append(node.right)
        return dict_ranges, originals

    # traverse tree, whenever we find const node we set the value of the node according to [name]. AST has 3 constant.
    def set_Num_value(values):
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
                    node.value = values[name]
                else:
                    node.value = values.pop(0)
            elif type(node) is Ite:
                q.append(node.condition)
                q.append(node.true_case)
                q.append(node.false_case)
            elif type(node) is Lt:
                q.append(node.left)
                q.append(node.right)
            # elif type(node) is AssignAction:
            #    q.append(node.value)
            elif type(node) is Addition:
                q.append(node.left)
                q.append(node.right)
        return

    def add_trajectory(self, inputs, actions):
        self.inputs = inputs
        self.actions = actions

    def evaluate(self, p):
        # self.actions --> produced by Oracle.
        # actions --> produced by AST.
        actions = get_action(self.inputs, p)
        actions_diff = spatial.distance.hamming(actions, np.array(self.actions))

        return 1-actions_diff

    def optimize(self, tree):
        self.tree = tree

        #gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 10}  # Optimizer configuration
        # list of Nums in the AST to optimize over
        list_Nums_range, originals = self.get_Num_range()
        print(list_Nums_range)
        bayesOpt = BayesianOptimization(self.evaluate, pbounds=list_Nums_range, verbose=0)

        try:
            # Bayesian Optimization
            bayesOpt.maximize(init_points=40, n_iter=10, kappa=2.5)
            # Update tree with optimized Nums
            self.set_Num_value(bayesOpt.max['params'])
            return bayesOpt.max['target']

        except Exception as error:
            print(error)
            self.set_Num_value(originals)
            return originals
        # utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        # for _ in range(5):
        #     next_point = bayesOpt.suggest(utility)
        #     bayesOpt.register(params=next_point)
        #
        #     print(next_point)
        # print(bayesOpt.max)

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