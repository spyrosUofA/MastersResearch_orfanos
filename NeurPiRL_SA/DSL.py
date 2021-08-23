import itertools
import numpy as np

class Node:
    def __init__(self):
        self.size = 1 # changed from 0

        # NEW >>
        self.number_children = 0
        self.current_child = 0
    
    def getSize(self):
        return self.size
    
    def toString(self):
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
                                observation_values):
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

        Observation.accepted_nodes = set(observation_values)
        Observation.accepted_types = [Observation.accepted_nodes]
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
                                    Addition.class_name(),
                                    Multiplication.class_name()])

        Multiplication.accepted_types = [Multiplication.accepted_nodes, Multiplication.accepted_nodes]

        Addition.accepted_nodes = set([Num.class_name(),
                                    Observation.class_name(),
                                    Addition.class_name(),
                                    Multiplication.class_name()])
        Addition.accepted_types = [Addition.accepted_nodes, Addition.accepted_nodes]


        # Do I need this? This is a new addition.
        #Observation.accepted_nodes = set([Num.class_name()])
        #Observation.accepted_nodes = set([Num.new(0), Num.new(1), Num.new(7)])
        Observation.accepted_nodes = set([0, 1, 2, 3, 4, 5, 6, 7])
        Observation.accepted_types = [Observation.accepted_nodes]

        # OK???
        #AssignAction.accepted_nodes = set([Num.class_name()])
        #AssignAction.accepted_nodes = set([Num.new(0), Num.new(1), Num.new(3)])
        AssignAction.accepted_nodes = set([0, 1, 2, 3])
        AssignAction.accepted_types = [AssignAction.accepted_nodes]

        Ite.accepted_nodes_bool = set([Lt.class_name()])
        Ite.accepted_nodes_block = set([AssignAction.class_name(), Addition.class_name(), Multiplication.class_name(), Ite.class_name()])
        Ite.accepted_types = [Ite.accepted_nodes_bool, Ite.accepted_nodes_block, Ite.accepted_nodes_block]

        Lt.accepted_nodes = set([Num.class_name(),
                                 Observation.class_name(),
                                 Addition.class_name(),
                                 Multiplication.class_name()])
        Lt.accepted_types = [Lt.accepted_nodes, Lt.accepted_nodes]

        Node.accepted_types = [set([Ite.class_name(), AssignAction.class_name()])]


    # <<< NEW

    @classmethod
    def name(cls):
        return cls.__name__

class StartSymbol(Node):
    def __init__(self):
        super(StartSymbol, self).__init__()
        self.size = 0
        self.number_children = 1

    @classmethod
    def new(cls, start):
        inst = cls()
        inst.add_child(start)
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


class Num_old(Node):
    def __init__(self, value):
        self.value = value
        self.size = 1

    def toString(self):
        return str(self.value)

    def interpret(self, env):
        return self.value

    def __eq__(self, other):
        if type(other) != Num:
            return False
        if self.value == other.value:
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


class Ite_old(Node):
    def __init__(self, condition, true_case, false_case):
        self.condition = condition
        self.true_case = true_case
        self.false_case = false_case
        
        self.size = condition.getSize() + true_case.getSize() + false_case.getSize() + 1
        
    def toString(self):
        return "(if" + self.condition.toString() + " then " + self.true_case.toString() + " else " + self.false_case.toString() + ")"
    
    def interpret(self, env):
        if self.condition.interpret(env):
            return self.true_case.interpret(env)
        else:
            return self.false_case.interpret(env)

    def __eq__(self, other):
        if type(other) != Ite:
            return False
        if self.condition == other.condition and self.true_case == other.true_case and self.false_case == other.false_case:
            return True
        return False

    def getBooleans(self):
        # BFS
        q = [self]
        bools = []

        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Ite:
                # Always a Lt expression
                bools.append(node.condition)

                # Add Lt expressions to boolean list
                if type(node.true_case) is Lt:
                    bools.append(node.true_case)
                if type(node.false_case) is Lt:
                    bools.append(node.false_case)

                # If true/false condition is Ite, add to q and keep traversing
                if type(node.true_case) is Ite:
                    q.append(node.true_case)
                if type(node.false_case) is Ite:
                    q.append(node.false_case)

            # Add Lt expressions to boolean list
            elif type(node) is Lt:
                bools.append(node)
        return bools

    def grow(plist, size):
        new_programs = []
        # defines the set of nodes accepted as conditions for an Ite
        accepted_condition_nodes = set([Lt.name()])
        # defines the set of nodes accepted as cases for an Ite
        accepted_case_nodes = set([AssignAction.name(), Ite.name()])

        combinations = list(itertools.product(range(1, size - 1), repeat=3))
        
        for c in combinations:
            
            if c[0] + c[1] + c[2] + 1 != size:
                continue
            
            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = plist.get_programs(c[0])
            program_set2 = plist.get_programs(c[1])
            program_set3 = plist.get_programs(c[2])
            
            for t1, programs1 in program_set1.items():                
                # skip if t1 isn't a node accepted as a condition node for Ite
                if t1 not in accepted_condition_nodes:
                    continue
                
                for p1 in programs1:
                    for t2, programs2 in program_set2.items():
                        # skip if t2 isn't a case node
                        if t2 not in accepted_case_nodes:
                            continue
                        
                        # p1 and all programs in programs2 satisfy constraints; grow the list
                        for p2 in programs2:
                            for t3, programs3 in program_set3.items():
                            # skip if t3 isn't a case node
                                if t3 not in accepted_case_nodes:
                                    continue
            
                                # produces a new program with Ite, p1, p2, and p3
                                for p3 in programs3:
                                    if p2 == p3:continue
                                    ite = Ite(p1, p2, p3)
                                    new_programs.append(ite)
                                    
                                    yield ite
        return new_programs


class AssignAction(Node):
    def __init__(self):
        super(AssignAction, self).__init__()
        self.number_children = 1

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)

        return inst

    def to_string(self):
        if len(self.children) == 0:
            raise Exception('AssignActionToReturn: Incomplete Program')

        return 'act = ' + self.children[0] #.to_string()

    def interpret(self, env):
        if len(self.children) == 0:
            raise Exception('AssignActionToReturn: Incomplete Program')

        #Levi: env['action_to_return'] = env['actions'][self.children[0].interpret(env)]
        #return env['action_to_return']

        #OLD: env['act'] = self.value.interpret(env)
        env['act'] = self.children[0] #.interpret(env)


class AssignAction_old(Node):
    def __init__(self, value):
        self.value = value
        self.size = self.value.getSize()

    def toString(self):
        return 'act = ' + str(self.value.toString())

    def interpret(self, env):
        env['act'] = self.value.interpret(env)

    def __eq__(self, other):
        if type(other) != AssignAction:
            return False
        if self.value == other.value:
            return True
        return False

    def grow(plist, size):
        #print("grow action")

        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Num.name()]) #, Observation.name(), Addition.name(), Multiplication.name()])

        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(range(1, size))  # [1], [1, 2], ..., [1, 2, 3, 4, 5, ..., 15]

        for c in combinations:
            # skip if the cost combination exceeds the limit
            if c + 1 != size:
                continue

            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = plist.get_programs(c)

            #print("p1", program_set1)

            # need this loop
            for t1, programs1 in program_set1.items():
                # skip if t1 isn't a node accepted by AA
                if t1 not in accepted_nodes:
                    continue

                # need this loop
                for p1 in programs1:
                    aa = AssignAction(p1)
                    new_programs.append(aa)

                    yield aa

        return new_programs


class Observation(Node):
    def __init__(self, index):
        self.index = index
        self.size = 1

        self.number_children = 1

    def toString(self):
        return 'obs[' + str(self.index) + ']'

    def to_string(self):
        return 'obs[' + str(self.index) + ']'

    def interpret(self, env):
        return env['obs'][self.index]

    def __eq__(self, other):
        if type(other) != Observation:
            return False
        if self.index == other.index:
            return True
        return False


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


class Addition_old(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        self.size = left.getSize() + right.getSize() + 1

    def toString(self):
        return "(" + self.left.toString() + " + " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) + self.right.interpret(env)

    def __eq__(self, other):
        if type(other) != Addition:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        if self.left == other.right and self.right == other.left:
            return True
        return False

    def grow(plist, size):
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Num.name(), Observation.name(), Multiplication.name(), ReLU.name()])

        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue

            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = plist.get_programs(c[0])
            program_set2 = plist.get_programs(c[1])

            for t1, programs1 in program_set1.items():
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue

                for p1 in programs1:
                    for t2, programs2 in program_set2.items():
                        # skip if t1 isn't a node accepted
                        if t2 not in accepted_nodes:
                            continue

                        if (t1 == Num.name() and t2 == Num.name()):
                            continue

                        # p1 and all programs in programs2 satisfy constraints; grow the list
                        for p2 in programs2:
                            st = Addition(p1, p2)
                            new_programs.append(st)

                            yield st
        return new_programs


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


class Multiplication_old(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        self.size = left.getSize() + right.getSize() + 1

    def toString(self):
        return "(" + self.left.toString() + " * " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) * self.right.interpret(env)

    def __eq__(self, other):
        if type(other) != Multiplication:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        if self.left == other.right and self.right == other.left:
            return True
        return False

    def grow(plist, size):
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Num.name(), Observation.name(), Addition.name(), ReLU.name()])

        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue

            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = plist.get_programs(c[0])
            program_set2 = plist.get_programs(c[1])

            for t1, programs1 in program_set1.items():
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue

                for p1 in programs1:
                    for t2, programs2 in program_set2.items():
                        # skip if t1 isn't a node accepted
                        if t2 not in accepted_nodes:
                            continue

                        #if (t1 == Observation.name() and t2 == Observation.name()):
                        #    continue

                        # p1 and all programs in programs2 satisfy constraints; grow the list
                        for p2 in programs2:
                            mp = Multiplication(p1, p2)
                            new_programs.append(mp)

                            yield mp
        return new_programs


class ReLU(Node):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

        self.size = 1

    def toString(self):
        return 'max(0, ' + str(self.weight) + " *dot* obs + " + str(self.bias) + ")"

    def interpret(self, env):
        #print(max(0.0, np.dot(self.weight, env['obs']) + self.bias))
        return max(0.0, np.dot(self.weight, env['obs']) + self.bias)

    def __eq__(self, other):
        if type(other) != ReLU:
            return False
        if (self.weight == other.weight).all() and (self.bias == other.bias).all():
            return True
        return False


Node.restore_original_production_rules()
