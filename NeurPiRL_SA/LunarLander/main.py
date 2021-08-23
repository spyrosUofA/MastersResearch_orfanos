from evaluation import *
import argparse
import numpy as np
import pandas as pd
import pickle
#from search.bottom_up_search import BottomUpSearch
from SimulatedAnnealing import SimulatedAnnealing
import copy
#from learning.iterated_best_response import IteratedBestResponse

def main():
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    parser = argparse.ArgumentParser()

    parser.add_argument('-search', action='store', dest='search_algorithm', 
                        default='SimulatedAnnealing', 
                        help='Search Algorithm (SimulatedAnnealing, BottomUpSearch, UCT)')
    
    parser.add_argument('-bound', action='store', dest='bound',
                        help='Bound for Bottom-Up Search')
    
    parser.add_argument('-e', action='store', dest='eval_function', 
                        default='Environment',
                        help='Environment, Imitation')
    
    parser.add_argument('-sim_function', action='store', dest='sim_function', 
                        default='Random', 
                        help='Simulation Function for UCT (Random or SA)')
     
    parser.add_argument('-n', action='store', dest='number_games', default=50,
                        help='Number of games played in each evaluation')
    
    parser.add_argument('-c', action='store', dest='uct_constant', default=1,
                        help='Constant value used in UCT search')
    
    parser.add_argument('-sims', action='store', dest='number_simulations', default=1,
                        help='Number of simulations used with UCT')
    
    parser.add_argument('-time', action='store', dest='time_limit', default=120,
                        help='Time limit in seconds')
    
    parser.add_argument('-temperature', action='store', dest='initial_temperature', default=100,
                        help='SA\'s initial temperature')
    
    parser.add_argument('-alpha', action='store', dest='alpha', default=0.6,
                        help='SA\'s alpha value')
    
    parser.add_argument('-beta', action='store', dest='beta', default=100,
                        help='SA\'s beta value')
    
    parser.add_argument('-log_file', action='store', dest='log_file',
                        help='File in which results will be saved')
    
    parser.add_argument('-program_file', action='store', dest='program_file',
                        help='File in which programs will be saved')
    
    parser.add_argument('--hole-node', action='store_true', default=False,
                        dest='use_hole_node',
                        help='Allow the use of hole nodes in the production rules of the AST.')
    
    parser.add_argument('--detect-equivalence', action='store_true', default=False,
                        dest='detect_equivalence',
                        help='Detect observational equivalence in Bottom-Up Search.')
    
    parser.add_argument('--triage', action='store_true', default=False,
                        dest='use_triage',
                        help='Use a 3-layer triage for evaluating programs.')
    
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        dest='use_bidirectional',
                        help='UCT simulations will use a library of programs generated with BUS.')
    
    parser.add_argument('-bidirectional_depth', action='store', dest='bidirectional_depth', default=1,
                        help='Maximum search depth for BUS when using UCT with bidirectional search')

    parser.add_argument('--double-programs', action='store_true', default=True,
                        dest='use_double_programs',
                        help='The program will have two instructions, one for yes-no decisions and another for column decisions')
    
    parser.add_argument('--iterated-best-response', action='store_true', default=False,
                        dest='run_ibr',
                        help='It will run Iterated Best Response')
    
    parser.add_argument('--reuse-tree', action='store_true', default=False,
                    dest='reuse_tree',
                    help='UCT reuses its tree and SA starts with previous solution in between iterations of IBR')


    parameters = parser.parse_args()
    number_games = int(parameters.number_games)
    number_simulations = int(parameters.number_simulations)
    eval_function = globals()[parameters.eval_function](number_games)
    time_limit = int(parameters.time_limit)
    algorithm = globals()[parameters.search_algorithm](parameters.log_file, parameters.program_file)

    # LOAD TRAJECTORY GIVEN BY NEURAL POLICY
    if parameters.eval_function == "Imitation":
        trajs = pd.read_csv("../LunarLander/trajectory.csv")
        observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]', 'o[4]', 'o[5]', 'o[6]', 'o[7]']].to_numpy()
        actions = trajs['a'].to_numpy()
        eval_function.add_trajectory(observations, actions)
        #print(trajs)

    if True:
        initial_program = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))

    if isinstance(algorithm, SimulatedAnnealing):
        
        from DSL import Ite, \
                                Lt, \
                                AssignAction, \
                                Addition, \
                                Multiplication, \
                                Observation, \
                                ReLU, \
                                Num
        
        terminals = [AssignAction]
        
        algorithm.search([AssignAction,
                          Ite,
                          Lt,
                          Num,
                          Observation,
                          ReLU,
                          Addition,
                          Multiplication],
                         [-0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
                         [0, 1, 2, 3, 4, 5, 6, 7],
                         [0, 1, 2, 3],
                         eval_function,
                         parameters.use_triage,
                         parameters.use_double_programs,
                         float(parameters.initial_temperature),
                         float(parameters.alpha),
                         float(parameters.beta),
                         time_limit,
                         None,
                         None)

if __name__ == "__main__":
    main()
