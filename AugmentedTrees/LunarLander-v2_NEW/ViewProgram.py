import numpy as np
from sklearn import tree
import pandas as pd
import re
import pickle




import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def get_rules(dtc, df):
    rules_list = []
    values_path = []
    values = dtc.tree_.value

    def RevTraverseTree(tree, node, rules, pathValues):
        '''
        Traverase an skl decision tree from a node (presumably a leaf node)
        up to the top, building the decision rules. The rules should be
        input as an empty list, which will be modified in place. The result
        is a nested list of tuples: (feature, direction (left=-1), threshold).
        The "tree" is a nested list of simplified tree attributes:
        [split feature, split threshold, left node, right node]
        '''
        # now find the node as either a left or right child of something
        # first try to find it as a left node

        try:
            prevnode = tree[2].index(node)
            leftright = '<='
            pathValues.append(values[prevnode])
        except ValueError:
            # failed, so find it as a right node - if this also causes an exception, something's really f'd up
            prevnode = tree[3].index(node)
            leftright = '>'
            pathValues.append(values[prevnode])

        # now let's get the rule that caused prevnode to -> node
        p1 = df.columns[tree[0][prevnode]]
        p2 = tree[1][prevnode]
        rules.append(str(p1) + ' ' + leftright + ' ' + str(p2))

        # if we've not yet reached the top, go up the tree one more step
        if prevnode != 0:
            RevTraverseTree(tree, prevnode, rules, pathValues)

    # get the nodes which are leaves
    leaves = dtc.tree_.children_left == -1
    leaves = np.arange(0,dtc.tree_.node_count)[leaves]

    # build a simpler tree as a nested list: [split feature, split threshold, left node, right node]
    thistree = [dtc.tree_.feature.tolist()]
    thistree.append(dtc.tree_.threshold.tolist())
    thistree.append(dtc.tree_.children_left.tolist())
    thistree.append(dtc.tree_.children_right.tolist())

    # get the decision rules for each leaf node & apply them
    for (ind,nod) in enumerate(leaves):

        # get the decision rules
        rules = []
        pathValues = []
        RevTraverseTree(thistree, nod, rules, pathValues)

        pathValues.insert(0, values[nod])
        pathValues = list(reversed(pathValues))

        rules = list(reversed(rules))

        rules_list.append(rules)
        values_path.append(pathValues)

    return (rules_list, values_path)


def relu_string(relus):
    relu_names = []
    for i, relu in enumerate(relus):
        name = '(' + str(np.around(relu[0], 2)) + " *dot* obs[:] + " + str(np.round(relu[1], 2)) + ")"
        relu_names.append(name)
    return relu_names








rews = np.load("Oracle/64x64/1/AugTreeRewards.npy").tolist()
relus = pickle.load(open("Oracle/64x64/1/ReLUs.pkl", "rb"))
#relu_names = relu_string(relus)
relu_names = ["w" + str(i).zfill(1) for i in range(64)]
relu_names.extend(["x", "y", "v_x", "v_y", "theta", "v_th", "c_l", "c_r"])

trees = pickle.load(open("Oracle/64x64/1/AugTreePrograms.pkl", "rb"))



regr_1 = trees[0]


print(regr_1[0], regr_1[1])

print(get_rules(regr_1, None))

print(regr_1.tree_.threshold)

print(regr_1.tree_)

exit()


tree_rules = tree.export_text(regr_1, feature_names=relu_names)

print(tree_rules)

print(relus[11])
print(relus[29])



# Extract decision rules as strings
decision_rules = tree_rules.replace("|--- class:", "act = ")
decision_rules = decision_rules.replace("|---", "if")
decision_rules = decision_rules.replace("|", "")

print(decision_rules)
boolean_rules = re.findall(r'if (.*)', decision_rules)





