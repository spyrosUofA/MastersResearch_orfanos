import copy
import pickle
from evaluation import Environment
import torch
import numpy as np
from ActorCritic.model import ActorCritic
from DSL import *
import gym


p1 = (Ite().new(Lt.new(Num.new(0.0), Num.new(0.0)), AssignAction.new(5), AssignAction.new(8)))


test_add = Addition.new(Num.new(2.0), Num.new(3.0))
test_mult = Multiplication.new(Num.new(0.0), Num.new(3.0))


def ite_same_same(p):
    if isinstance(p, Ite):
        if isinstance(p.children[1], AssignAction) and isinstance(p.children[2], AssignAction):
            if p.children[1].children[0] == p.children[2].children[0]:
                p = p.children[2]
    return p


def num_lt_num(p):
    
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
                p = Num.new(p.children[1].children[0])
        elif isinstance(p.children[1], Num):
            if p.children[1].children[0] == 0:
                p = Num.new(p.children[0].children[0])
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


def simplify_program(p):
    program = copy.deepcopy(p)
    program = ite_same_same(program)
    program = plus_zero(program)
    program = num_plus_num(program)
    program = num_times_num(program)
    program = num_lt_num(program)
    return program

