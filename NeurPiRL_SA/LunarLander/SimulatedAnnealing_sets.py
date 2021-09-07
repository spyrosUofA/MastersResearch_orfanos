from DSL import *
import numpy as np
import random
import time
from os.path import join
import os
import pickle
import copy

np.set_printoptions(precision=2)
random.seed(0)

class SimulatedAnnealing():

    def __init__(self, folder_name, log_file, program_file, seed=0):
        ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))

        self.log_folder = 'logs/' + folder_name + '/'
        self.program_folder = 'programs/' + folder_name + '/'
        self.binary_programs = 'binary_programs/' + folder_name + '/'

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        if not os.path.exists(self.program_folder):
            os.makedirs(self.program_folder)

        if not os.path.exists(self.binary_programs):
            os.makedirs(self.binary_programs)

        self.log_file = 'sa-' + str(ncpus) + '-cpus' + log_file
        self.program_file = 'sa-' + str(ncpus) + '-cpus' + program_file
        self.binary_program_file = self.binary_programs + 'sa-' + str(ncpus) + '-cpus' + program_file + '.pkl'

        # Set seed
        #np.random.seed(seed)
        #random.seed(seed)

    def mutate_inner_nodes_ast(self, p, index):
        self.processed += 1

        if not isinstance(p, Node):
            return False

        for i in range(p.get_number_children()):

            if index == self.processed:
                # Accepted rules for the i-th child
                types = p.accepted_rules(i)

                # Generate instance of a random accepted rule
                if isinstance(p, AssignAction) or isinstance(p, Observation) or isinstance(p, Num):
                    child = list(types)[random.randrange(len(types))]
                elif isinstance(p, ReLU):
                    types = ReLU.accepted_types
                    child = list(types)[random.randrange(len(types))]
                else:
                    child = Node.factory(list(types)[random.randrange(len(types))])

                # Randomly generate the child
                if isinstance(child, Node):
                    self.fill_random_program(child, 0, 4)

                # Replacing previous child with the randomly generated one
                p.replace_child(child, i)
                return True

            #print(i, " ", index, p.children[i])
            mutated = self.mutate_inner_nodes_ast(p.children[i], index)

            if mutated:

                # Fixing the size of all nodes in the AST along the modified branch 
                modified_size = 1
                for j in range(p.get_number_children()):
                    if isinstance(p.children[j], Node):
                        modified_size += p.children[j].get_size()
                    else:
                        modified_size += 1
                p.set_size(modified_size)

                return True

        return False

    def mutate(self, p):
        try:
            index = random.randrange(p.get_size())
        except:
            index = 0

        # Mutating the root of the AST
        if index == 0:

            if self.use_double_program:
                p = StartSymbol()
            else:
                initial_types = Node.accepted_rules(0)
                p = Node.factory(list(initial_types)[random.randrange(len(initial_types))])
            self.fill_random_program(p, self.initial_depth_ast, self.max_mutation_depth)

            return p

        self.processed = 0
        self.mutate_inner_nodes_ast(p, index)

        return p

    def return_terminal_child(self, p, types):
        terminal_types = []

        for t in types:
            child = p.factory(t)

            if child.get_number_children() == 0 or isinstance(child, Num) or isinstance(child, Observation) or isinstance(
                    child, AssignAction) or isinstance(child, ReLU):
                terminal_types.append(child)

        if len(terminal_types) == 0:
            for t in types:
                child = p.factory(t)

                if child.get_number_children() == 1:
                    terminal_types.append(child)

        if len(terminal_types) > 0:
            return terminal_types[random.randrange(len(terminal_types))]

        return p.factory(list(types)[random.randrange(len(types))])

    def fill_random_program(self, p, depth, max_depth):

        size = p.get_size()

        for i in range(p.get_number_children()):

            print("Current Child #: ", i)
            print("Current type: ", type(p))
            types = p.accepted_rules(i)
            print("Accepted types: ", types)

            if isinstance(p, AssignAction) or isinstance(p, Observation) or isinstance(p, Num):
                types = p.accepted_rules(i)
                child = list(types)[random.randrange(len(types))]
                p.add_child(child)
                size += 1

            elif isinstance(p, ReLU):
                types = ReLU.accepted_types
                #print(types)
                child = list(types)[random.randrange(len(types))]
                p.add_child(child)

                size += 1

            elif depth >= max_depth:
                types = p.accepted_rules(i)
                child = self.return_terminal_child(p, types)
                p.add_child(child)
                child_size = self.fill_random_program(child, depth + 1, max_depth)

                size += child_size
            else:
                types = p.accepted_rules(i)
                some_num = random.randrange(len(types))
                child = p.factory(list(types)[some_num])
                #child = p.factory(list(types)[random.randrange(len(types))])
                print("Child", some_num, " ", child)
                p.add_child(child)
                child_size = self.fill_random_program(child, depth + 1, max_depth)

                size += child_size

        p.set_size(size)
        return size

    def random_program(self):
        if self.use_double_program:
            p = StartSymbol()
        else:
            initial_types = list(Node.accepted_initial_rules()[0])
            p = Node.factory(initial_types[random.randrange(len(initial_types))])

        self.fill_random_program(p, self.initial_depth_ast, self.max_mutation_depth)

        return p

    def accept_function(self, current_score, next_score):
        return np.exp(self.beta * (next_score - current_score) / self.current_temperature)

    def decrease_temperature(self, i):
        #         self.current_temperature = self.initial_temperature * self.alpha ** i
        self.current_temperature = self.initial_temperature / (1 + self.alpha * (i))

    def search(self,
               operations,
               numeric_constant_values,
               observation_values,
               action_values,
               eval_function,
               use_triage,
               use_double_program,
               initial_temperature,
               alpha,
               beta,
               time_limit,
               winrate_target=None,
               initial_program=None,
               bayes_opt=False):

        np.random.seed(0)
        random.seed(0)

        time_start = time.time()

        self.winrate_target = winrate_target
        self.use_double_program = use_double_program

        Node.filter_production_rules(operations,
                                     numeric_constant_values,
                                     observation_values,
                                     action_values)

        self.max_mutation_depth = 4
        self.initial_depth_ast = 0
        self.initial_temperature = initial_temperature
        self.alpha = alpha
        self.beta = beta
        self.slack_time = 600

        Num.accepted_types = [set(numeric_constant_values)]
        AssignAction.accepted_types = [set(action_values)]
        Observation.accepted_types = [set(observation_values)]
        ReLU.accepted_types = pickle.load(open("ReLU_accepted_nodes.pickle", "rb"))

        self.operations = operations
        self.numeric_constant_values = numeric_constant_values
        self.eval_function = eval_function

        best_score = self.eval_function.worst_score
        best_program = None

        id_log = 1
        number_games_played = 0

        if initial_program is not None:
            current_program = copy.deepcopy(initial_program)
        else:
            current_program = self.random_program()

        print(current_program.to_string())
        exit()

        while True:
            # is this the right spot?
            if bayes_opt:
                self.eval_function.optimize(current_program)

            self.current_temperature = self.initial_temperature

            if use_triage:
                current_score, number_matches_played = self.eval_function.eval_triage(current_program, best_score)
            else:
                current_score, number_matches_played = self.eval_function.evaluate(current_program)

            number_games_played += number_matches_played

            iteration_number = 1

            if best_program is None or current_score > best_score:
                best_score = current_score
                best_program = current_program

                with open(self.binary_program_file, 'wb') as file_program:
                    pickle.dump(current_program, file_program)

                if self.winrate_target is None:
                    with open(join(self.log_folder + self.log_file), 'a') as results_file:
                        results_file.write(("{:d}, {:f}, {:d}, {:f} \n".format(id_log,
                                                                               best_score,
                                                                               number_games_played,
                                                                               time.time() - time_start)))

                    with open(join(self.program_folder + self.program_file), 'a') as results_file:
                        results_file.write(("{:d} \n".format(id_log)))
                        results_file.write(best_program.to_string())
                        results_file.write('\n')

                    id_log += 1

            while self.current_temperature > 1:

                time_end = time.time()

                if time_end - time_start > time_limit - self.slack_time:
                    if self.winrate_target is None:
                        with open(join(self.log_folder + self.log_file), 'a') as results_file:
                            results_file.write(("{:d}, {:f}, {:d}, {:f} \n".format(id_log,
                                                                                   best_score,
                                                                                   number_games_played,
                                                                                   time_end - time_start)))
                    return best_score, best_program

                copy_program = copy.deepcopy(current_program)

                #print('\nCurrent: ')
                #print(current_program.to_string())
                mutation = self.mutate(copy_program)
                #print('Mutated: ')
                #print(mutation.to_string(), '\n')

                # is this the right spot?
                if bayes_opt:
                    self.eval_function.optimize(current_program)

                if use_triage:
                    next_score, number_matches_played = self.eval_function.eval_triage(mutation, best_score)
                else:
                    next_score, number_matches_played = self.eval_function.evaluate(mutation)

                if self.winrate_target is not None and next_score >= self.winrate_target:
                    return next_score, mutation

                number_games_played += number_matches_played

                if best_program is None or next_score > best_score:

                    best_score = next_score
                    best_program = mutation

                    print('\nCurrent best: ', best_score)
                    print(best_program.to_string())

                    with open(self.binary_program_file, 'wb') as file_program:
                        pickle.dump(current_program, file_program)

                    if self.winrate_target is None:
                        with open(join(self.log_folder + self.log_file), 'a') as results_file:
                            results_file.write(("{:d}, {:f}, {:d}, {:f} \n".format(id_log,
                                                                                   best_score,
                                                                                   number_games_played,
                                                                                   time_end - time_start)))

                        with open(join(self.program_folder + self.program_file), 'a') as results_file:
                            results_file.write(("{:d} \n".format(id_log)))
                            results_file.write(best_program.to_string())
                            results_file.write('\n')

                        id_log += 1

#                print('\nCurrent best: ', best_score)
#                print(best_program.to_string())

                prob_accept = min(1, self.accept_function(current_score, next_score))

                prob = random.uniform(0, 1)
                if prob < prob_accept:
                    #                     print('Probability of accepting: ', prob_accept, next_score, current_score, self.current_temperature)

                    #                     print(mutation.to_string())
                    #                     print('Score: ', next_score)
                    #                     print()

                    current_program = mutation
                    current_score = next_score

                #                     print('Current score: ', current_score)

                iteration_number += 1

                self.decrease_temperature(iteration_number)
            #                 print('Current Temp: ', self.current_temperature)

            if initial_program is not None:
                current_program = copy.deepcopy(initial_program)
            else:
                if best_score == self.eval_function.worst_score:
                    current_program = self.random_program()
                else:
                    current_program = copy.deepcopy(best_program)

        return best_score, best_program
