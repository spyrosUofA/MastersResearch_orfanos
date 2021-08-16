import itertools
import numpy as np

class Node:
    def __init__(self):
        self.size = 1 # changed from 0

        # NEW >>
        self.number_children = 0
        self.current_child = 0

        self.children = []

    def getSize(self):
        return self.size
    
    def to_string(self):
        raise Exception('Unimplemented method: toString')
    
    def interpret(self):
        raise Exception('Unimplemented method: interpret')
    
    def grow(self, plist, new_plist):
        pass

    # NEW >>>
    def add_child(self, child):
        if len(self.children) + 1 > self.number_children:
            raise Exception('Unsupported number of children')

        self.children.append(child)
        self.current_child += 1

        if child is None or not isinstance(child, Node):
            self.size += 1
        else:
            self.size += child.size

    def get_current_child(self):
        return self.current_child

    def get_number_children(self):
        return self.number_children

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def replace_child(self, child, i):

        if len(self.children) < i + 1:
            self.add_child(child)
        else:
            if isinstance(self.children[i], Node):
                self.size -= self.children[i].size
            else:
                self.size -= 1

            if isinstance(child, Node):
                self.size += child.size
            else:
                self.size += 1

            self.children[i] = child

    @classmethod
    def accepted_rules(cls, child):
        return cls.accepted_types[child]

    @classmethod
    def class_name(cls):
        return cls.__name__

    @staticmethod
    def factory(classname):
        if classname not in globals():
            return classname
        return globals()[classname]()

    @classmethod
    def accepted_initial_rules(cls):
        return cls.accepted_types

    @staticmethod
    def filter_production_rules(operations,
                                numeric_constant_values,
                                observation_values,
                                action_values):
        rules = set()
        for op in operations:
            rules.add(op.class_name())

        #for func in functions_scalars:
        #    rules.add(func.class_name())

        if len(numeric_constant_values) > 0:
            rules.add(Num.class_name())

        if len(observation_values) > 0:
            rules.add(Observation.class_name())

        # NEW >>>>
        #AssignAction.accepted_nodes = set(action_values)
        #AssignAction.accepted_types = [AssignAction.accepted_nodes]

        #Observation.accepted_nodes = set(observation_values)
        #Observation.accepted_types = [Observation.accepted_nodes]
        # <<< NEW

        rules.add(None)

        list_all_productions = [Node,
                                Ite,
                                Lt,
                                Addition,
                                Multiplication]

        for op in list_all_productions:

            op_to_remove = []

            for types in op.accepted_types:
                for op in types:
                    if op not in rules:
                        op_to_remove.append(op)

                for op in op_to_remove:
                    if op in types:
                        types.remove(op)

    @staticmethod
    def restore_original_production_rules():
        StartSymbol.accepted_nodes = set([Ite.class_name(), AssignAction.class_name()])
        StartSymbol.accepted_types = [StartSymbol.accepted_nodes]

        Multiplication.accepted_nodes = set([Num.class_name(),
                                    Observation.class_name(),
                                    ReLU.class_name(),
                                    Addition.class_name(),
                                    Multiplication.class_name()])
        Multiplication.accepted_types = [Multiplication.accepted_nodes, Multiplication.accepted_nodes]

        Addition.accepted_nodes = set([Num.class_name(),
                                    Observation.class_name(),
                                    ReLU.class_name(),
                                    Addition.class_name(),
                                    Multiplication.class_name()])
        Addition.accepted_types = [Addition.accepted_nodes, Addition.accepted_nodes]

        Ite.accepted_nodes_bool = set([Lt.class_name()])
        Ite.accepted_nodes_block = set([AssignAction.class_name(), Ite.class_name()])
        Ite.accepted_types = [Ite.accepted_nodes_bool, Ite.accepted_nodes_block, Ite.accepted_nodes_block]

        Lt.accepted_nodes = set([Num.class_name(),
                                 Observation.class_name(),
                                 ReLU.class_name(),
                                 Addition.class_name(),
                                 Multiplication.class_name()])
        Lt.accepted_types = [Lt.accepted_nodes, Lt.accepted_nodes]

        Node.accepted_types = [set([Ite.class_name(), AssignAction.class_name()])]

    @classmethod
    def name(cls):
        return cls.__name__


class StartSymbol(Node):
    def __init__(self):
        super(StartSymbol, self).__init__()
        self.size = 0
        self.number_children = 1

    @classmethod
    def new(cls, yes_no):
        inst = cls()
        inst.add_child(yes_no)
        return inst

    def to_string(self):
        return self.children[0].to_string()

    def interpret(self, env):
        return self.children[0].interpret(env)


class Num(Node):

    def __init__(self):
        super(Num, self).__init__()
        self.number_children = 1
        self.size = 0

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)

        return inst

    def to_string(self):
        if len(self.children) == 0:
            raise Exception('VarScalar: Incomplete Program')

        return str(self.children[0])

    def interpret(self, env):
        if len(self.children) == 0:
            raise Exception('VarScalar: Incomplete Program')

        return self.children[0]


class AssignAction(Node):
    def __init__(self):
        super(AssignAction, self).__init__()
        self.number_children = 1
        self.size = 0

    def to_string(self):
        if len(self.children) == 0:
            raise Exception('AssignAction: Incomplete Program')

        return 'act = ' + str(self.children[0])

    def interpret(self, env):
        if len(self.children) == 0:
            raise Exception('AssignAction: Incomplete Program')

        env['act'] = self.children[0]

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)

        return inst


class Observation(Node):
    def __init__(self):
        super(Observation, self).__init__()
        self.number_children = 1 #?
        self.size = 0 #?

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)

        return inst

    def to_string(self):
        if len(self.children) == 0:
            raise Exception('Observation: Incomplete Program')

        return 'obs[' + str(self.children[0]) + ']'

    def interpret(self, env):
        if len(self.children) == 0:
            raise Exception('Observation: Incomplete Program')

        return env['obs'][self.children[0]]

    def __eq__(self, other):
        if type(other) != Observation:
            return False
        if self.index == other.index:
            return True
        return False


class ReLU(Node):
    def __init__(self):
        super(ReLU, self).__init__()
        self.number_children = 1
        self.size = 0

    @classmethod
    def new(cls, weight_bias):
        inst = cls()
        inst.add_child(weight_bias)
        #inst.add_child(bias)

        return inst

    def to_string(self):
        return 'max(0, ' + str(np.around(self.children[0][0], 3)) + " *dot* obs[:] + " + str(np.round(self.children[0][1], 3)) + ")"
        #return 'max(0, ' + str(self.children[0][0]) + " *dot* obs[:] + " + str(self.children[0][1]) + ")"

    def interpret(self, env):
        return max(0.0, np.dot(self.children[0][0], env['obs']) + self.children[0][1])

    def __eq__(self, other):
        if type(other) != ReLU:
            return False
        if (self.weight == other.weight).all() and (self.bias == other.bias).all():
            return True
        return False


class Lt(Node):
    def __init__(self):
        super(Lt, self).__init__()
        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " < " + self.children[1].to_string() + ")"

    def interpret(self, env):
        return self.children[0].interpret(env) < self.children[1].interpret(env)


class Ite(Node):
    def __init__(self):
        super(Ite, self).__init__()

        self.number_children = 3

    @classmethod
    def new(cls, bool_expression, true_block, false_block):
        inst = cls()
        inst.add_child(bool_expression)
        inst.add_child(true_block)
        inst.add_child(false_block)

        return inst

    def to_string(self):
        return '(if ' + self.children[0].to_string() + ' then: ' + self.children[1].to_string() + ' else: ' + \
               self.children[2].to_string() + ")"

    def interpret(self, env):
        if self.children[0].interpret(env):
            return self.children[1].interpret(env)
        else:
            return self.children[2].interpret(env)


class Addition(Node):
    def __init__(self):
        super(Addition, self).__init__()

        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " + " + self.children[1].to_string() + ")"

    def interpret(self, env):
        return self.children[0].interpret(env) + self.children[1].interpret(env)


class Multiplication(Node):
    def __init__(self):
        super(Multiplication, self).__init__()

        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        if len(self.children) < 2:
            raise Exception('Times: Incomplete Program')

        return "(" + self.children[0].to_string() + " * " + self.children[1].to_string() + ")"

    def interpret(self, env):
        if len(self.children) < 2:
            raise Exception('Times: Incomplete Program')

        return self.children[0].interpret(env) * self.children[1].interpret(env)


Node.restore_original_production_rules()


