import random
import copy
import numpy as np
import gym

class Strategy:
    def __init__(self):
        self.ranks = []

    def mutate(self):
        index = random.randint(0, len(self.ranks))

        chop_index = random.randint(0, len(self.ranks))
        self.ranks = []
        for i in range(chop_index + 1):
            self.ranks.append(i)

        while True:
            prob = random.random()

            if prob < 0.5:
                if len(self.ranks) > 0:
                    self.ranks.append(self.ranks[-1] + 1)
                else:
                    self.ranks.append(0)
            else:
                break



def evaluate(p, episode_count=25):
    steps = 0
    averaged = 0

    env = gym.make('LunarLander-v2')

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

            if done:
                break
        averaged += reward

    return averaged / episode_count


def SA(initial_temperature, beta, alpha, initial_program):
    current_program = initial_program

    current_temperature = initial_temperature

    current_score = evaluate(current_program)

    iteration_number = 1
    best_program = current_program
    best_score = current_score

    while current_temperature > 1:
        mutation = copy.deepcopy(current_program)
        mutation.mutate()

        next_score = evaluate(mutation)

        if next_score > best_score:
            best_score = next_score
            best_program = mutation

            if best_score == 1:
                return best_program, best_score

        prob_accept = min(1, np.exp(beta * (next_score - current_score) / current_temperature))

        prob = random.uniform(0, 1)
        if prob < prob_accept:
            current_program = mutation
            current_score = next_score

        iteration_number += 1

        current_temperature = initial_temperature / (1 + alpha * (iteration_number))
    return best_program, best_score


def IBR():
    s = Strategy()
    for _ in range(10):
        s.mutate()

    evaluation_ibr = EvalIBR(s)
    p = s
    for i in range(3000):
        evaluation_ibr = EvalIBR(s)
        p, score = SA(3000, 1000, 0.9, evaluation_ibr, p)

        if score == 1:
            s = p
            evaluation_ibr = EvalIBR(s)

            print(i, len(p.ranks))
    return s


def FP():
    s = Strategy()
    for _ in range(10):
        s.mutate()

    evaluation_ibr = EvalFP(s)
    p = s
    for i in range(3000):
        p, score = SA(3000, 1000, 0.9, evaluation_ibr, s)

        if score == 1:
            s = p
            evaluation_ibr.add_opponent(p)

            print(i, len(p.ranks), score)
    return p


for _ in range(1):
    p = IBR()
    s = FP()

    print(len(p.ranks), len(s.ranks))