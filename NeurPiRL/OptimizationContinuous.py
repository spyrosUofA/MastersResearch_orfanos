from bayes_opt import BayesianOptimization, UtilityFunction
from scipy import spatial
import numpy as np
from NeurPiRL.DSL import Num, Ite, Lt, AssignAction, Addition, ReLU
#from BottomUpSearch import get_action
import copy

def create_interval(value, delta):
    interval = (value - delta, value + delta)
    return interval


def get_action(obs, p):
    actions = []
    for ob in obs:
        namespace = {'obs': ob, 'act': 0}
        p.interpret(namespace)
        actions.append(namespace['act'])
    return actions


class ParameterFinderContinuous():
    def __init__(self, inputs, actions):
        #all observations
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
            if type(node) is Num:
                name = "Num"+str(i)
                i+=1
                originals.append(node.value)
                interval = create_interval(node.value, 0.1)
                dict_ranges[name] = copy.deepcopy(interval)
                #print(type(interval))
            elif type(node) is Ite:
                q.append(node.condition)
                q.append(node.true_case)
                q.append(node.false_case)
            elif type(node) is Lt:
                q.append(node.left)
                q.append(node.right)
            elif type(node) is AssignAction:
                q.append(node.value)
            elif type(node) is Addition:
                q.append(node.left)
                q.append(node.right)
        return dict_ranges, originals

    # traverse tree, whenever we find const node we set the value of the node according to [name]. AST has 3 constant.
    def set_Num_value(self, values):
        # BFS
        q = []
        i=1
        q.append(self.tree)
        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Num:
                name = "Num"+str(i)
                i+=1
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
            # does this belong?
            elif type(node) is AssignAction:
                q.append(node.value)
            elif type(node) is Addition:
                q.append(node.left)
                q.append(node.right)
        return

    def find_distance(self, **kwargs):
        numNodes = np.fromiter(kwargs.values(), dtype=float)
        self.set_Num_value(numNodes.tolist())
        actions = get_action(self.inputs, self.tree)
        actions_diff = spatial.distance.euclidean(actions, np.array(self.actions))

        #print(actions_diff)
        # self.actions --> from the trajectory.
        # actions --> learned AST.

        diff_total = -(actions_diff)/float(len(self.actions))

        return diff_total

    def optimize(self, tree):
        self.tree = tree
        #gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 10}  # Optimizer configuration
        list_Nums_range, originals = self.get_Num_range()
        #print(list_Nums_range)
        bayesOpt = BayesianOptimization(self.find_distance,
                                        pbounds=list_Nums_range, verbose=0)
        try:
            #bayesOpt.maximize(init_points=5, n_iter=10, kappa=5, **gp_params)
            bayesOpt.maximize(init_points=40, n_iter=10, kappa=2.5)
            #print(bayesOpt.max['params'])
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

    def find_distance_neuron(self, **kwargs):
        #from BottomUpSearch_so import get_action
        numNodes = np.fromiter(kwargs.values(), dtype=float)
        self.set_Num_value(numNodes.tolist())

        # actions --> produced by AST
        actions = get_action(self.inputs, self.tree)

        # self.actions --> given by trajectory
        actions_diff = spatial.distance.euclidean(actions, np.array(self.actions))
        diff_total = -(actions_diff)/float(len(self.actions))

        #print(diff_total)

        return diff_total

    def optimize_neuron(self, tree):
        self.tree = tree
        #gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 10}  # Optimizer configuration
        list_Nums_range, originals = self.get_Num_range()

        #print(list_Nums_range)
        #print(self.tree.toString())

        #print(list_Nums_range) # (originals-0.1, originals+0.1)
        #print(originals)  # originals

        bayesOpt = BayesianOptimization(self.find_distance_neuron, pbounds=list_Nums_range, verbose=0)

        try:
            #bayesOpt.maximize(init_points=5, n_iter=10, kappa=5, **gp_params)
            bayesOpt.maximize(init_points=25, n_iter=5, kappa=2.5)
            self.set_Num_value(bayesOpt.max['params'])
            #print("passed!!")
            #print(bayesOpt.max)

            return bayesOpt.max['target']
        except Exception as error:
            #print(bayesOpt.max['params'])
            #print(list_Nums_range)
            print("Error!!!", error)
            self.set_Num_value(originals)
            return -100 #originals

        # utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        # for _ in range(5):
        #     next_point = bayesOpt.suggest(utility)
        #     bayesOpt.register(params=next_point)
        #
        #     print(next_point)
        # print(bayesOpt.max)
