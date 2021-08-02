from evaluation import Evaluate
import argparse
import numpy as np
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
                        default='Evaluate',
                        help='Evaluation function (EvalDoubleProgramDefeatsStrategy, EvalImitationAgent, EvalSingleProgramDefeatsStrategy, EvalColumnActionDefeatsStrategy, or EvalYesNoActionDefeatsStrategy)')
    
    parser.add_argument('-sim_function', action='store', dest='sim_function', 
                        default='Random', 
                        help='Simulation Function for UCT (Random or SA)')
     
    parser.add_argument('-n', action='store', dest='number_games',
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

    parameters.number_games = 5
    
    number_games = int(parameters.number_games)
    number_simulations = int(parameters.number_simulations)
    eval_function = globals()[parameters.eval_function](number_games)
    
    time_limit = int(parameters.time_limit)
    algorithm = globals()[parameters.search_algorithm](parameters.log_file, parameters.program_file)
    uct_constant = float(parameters.uct_constant)
    bidirectional_depth = int(parameters.bidirectional_depth)

        

    if isinstance(algorithm, SimulatedAnnealing):
        
        from DSL import Ite, \
                                Lt, \
                                AssignAction, \
                                Addition, \
                                Multiplication, \
                                Observation, \
                                Num
        
        terminals = [AssignAction]
        
        algorithm.search([Ite,
                         Lt,
                         Num,
                         Observation,
                         AssignAction],
                            [0.236, 3.232],
                            [0, 1, 2, 3, 4, 5, 6, 7],
                            [0, 1, 2, 3],
                            #[],
                            #['neutrals', 'actions's],
                            #['progress_value', 'move_value'],
                            #terminals,
                            eval_function,
                            parameters.use_triage,
                            parameters.use_double_programs,
                            float(parameters.initial_temperature),
                            float(parameters.alpha),
                            float(parameters.beta),
                            time_limit)

if __name__ == "__main__":
    main()
