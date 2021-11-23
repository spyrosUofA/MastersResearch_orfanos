import copy
import pickle

def ite_same_same(p):
    if isinstance(p, Ite):
        if isinstance(p.children[1], AssignAction) and isinstance(p.children[2], AssignAction):
            if p.children[1].children[0] == p.children[2].children[0]:
                p = p.children[2]
    return p


def num_lt_num(p):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], Num) and isinstance(child.children[1], Num):
                if child.children[0].children[0] < child.children[1].children[0]:
                    p = p.children[1]
                else:
                    p = p.children[2]
    return p


def num_plus_num(p):
    if isinstance(p, Addition):
        if isinstance(p.children[0], Num) and isinstance(p.children[1], Num):
            p = Num.new(p.children[0].children[0] + p.children[1].children[0])
    return p


def num_times_num(p):
    if isinstance(p, Multiplication):
        if isinstance(p.children[0], Num) and isinstance(p.children[1], Num):
            p = Num.new(p.children[0].children[0] * p.children[1].children[0])
    return p


def plus_zero(p):
    if isinstance(p, Addition):
        if isinstance(p.children[0], Num):
            if p.children[0].children[0] == 0:
                p = p.children[1]
        elif isinstance(p.children[1], Num):
            if p.children[1].children[0] == 0:
                p = p.children[0]
    return p


def times_zero(p):
    if isinstance(p, Multiplication):
        if isinstance(p.children[0], Num):
            if p.children[0].children[0] == 0:
                p = Num.new(0.0)
        elif isinstance(p.children[1], Num):
            if p.children[1].children[0] == 0:
                p = Num.new(0.0)
    return p


def relu_lt_0(p):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], ReLU) and isinstance(child.children[1], Num):
                if child.children[1].children[0] <= 0:
                    p = p.children[2] # false case
    return p


def negative_lt_relu(p):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], Num) and isinstance(child.children[1], ReLU):
                if child.children[0].children[0] < 0:
                    p = p.children[1] # true case
    return p

def simplify_node(p):
    program = copy.deepcopy(p)
    program = ite_same_same(program)
    program = times_zero(program)
    program = plus_zero(program)
    program = num_plus_num(program)
    program = num_times_num(program)
    program = num_lt_num(program)
    program = relu_lt_0(program)
    program = negative_lt_relu(program)
    return program


def simplify_program(p):
    for i in range(p.get_number_children()):
        simplified = simplify_node(p.children[i])
        p.replace_child(simplified, i)
        # simplify children
        if isinstance(p.children[i], Node):
            simplify_program(p.children[i])
    return p


def simplify_program1(p):
    while True:
        p_next = simplify_program(p)
        if p_next is p:

            print(p_next, p)
            return p
        else:
            p = p_next



from DSL import *
from evaluation import *
import time

#policy = pickle.load(open("../binary_programs/D000/Oracle-15/sa_cpus-16_n-25_c-None_run-8.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-13/sa_cpus-16_n-25_c-None_run-3.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-12/sa_cpus-16_n-25_c-None_run-3.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-4/sa_cpus-16_n-25_c-None_run-25.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-9/sa_cpus-16_n-25_c-None_run-9.pkl", "rb"))

#policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-8/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))
policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-13/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))
print(policy.to_string() + '\n')
p_new = simplify_program1(policy)
print(p_new.to_string() + '\n')
print("------------------")

policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-13/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))

print(policy.to_string() + '\n')
policy = simplify_program(policy)
print(policy.to_string() + '\n')
policy = simplify_program(policy)
print(policy.to_string() + '\n')
policy = simplify_program(policy)
print(policy.to_string() + '\n')
policy = simplify_program(policy)
print(policy.to_string() + '\n')

exit()


rew = 0
for i in range(1, 16, 1):
    policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-" + str(i) + "/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))

    print(policy.to_string() + '\n')
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    print(policy.to_string() + '\n')

    rew += Environment({}, 200, 1).evaluate(policy)

print(rew / 15.0)
# E010_D010_old: 163.48063343676603

