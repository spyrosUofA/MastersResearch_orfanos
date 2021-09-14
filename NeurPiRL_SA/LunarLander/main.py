import argparse
import pickle
from SimulatedAnnealing import SimulatedAnnealing
from evaluation import Environment, Imitation, DAgger

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', action='store', dest='seed', default=0,
                        help='Set seed for reproducible results.')

    parser.add_argument('-search', action='store', dest='search_algorithm', 
                        default='SimulatedAnnealing', 
                        help='Search Algorithm (SimulatedAnnealing, BottomUpSearch, UCT)')
    
    parser.add_argument('-bound', action='store', dest='bound',
                        help='Bound for Bottom-Up Search')
    
    parser.add_argument('-e', action='store', dest='eval_function',
                        default='Environment',
                        help='Environment, Imitation, DAgger')

    parser.add_argument('--bo', action='store_true', dest='bayes_opt', default=False,
                        help='Bayesian Optimization toggle (default=False)')

    parser.add_argument('-ip', action='store', dest='init_program', default=None,
                        help='Initial Program (default=None)')

    parser.add_argument('-ocl', action='store', dest='oracle', default=None,
                        help='Oracle directory containing: Policy.pth, ReLUs.pkl, Actions.npy, Observations.npy')

    parser.add_argument('--relu', action='store', dest='relu_augment', default=None,
                        help='Path to .pkl file of ReLU_programs to augment the DSL (default=None)')
     
    parser.add_argument('-n', action='store', dest='number_games', default=25,
                        help='Number of games played in each evaluation')
    
    parser.add_argument('-time', action='store', dest='time_limit', default=1200,
                        help='Time limit in seconds')
    
    parser.add_argument('-temperature', action='store', dest='initial_temperature', default=100,
                        help='SA\'s initial temperature')
    
    parser.add_argument('-alpha', action='store', dest='alpha', default=0.6,
                        help='SA\'s alpha value')
    
    parser.add_argument('-beta', action='store', dest='beta', default=100,
                        help='SA\'s beta value')

    parser.add_argument('-file_name', action='store', dest='file_name', default="",
                        help='File in which results will be saved')
    
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

    # Specify folder
    folder_name = "Eval-" + str(parameters.eval_function) + "_BayesOpt-" + str(parameters.bayes_opt) +  \
                  "_ReLU-" + str(parameters.relu_augment is not None) + "_InitProg-" + str(parameters.init_program is not None)

    # Casting to integers
    number_games = int(parameters.number_games)
    seed = int(parameters.seed)
    time_limit = int(parameters.time_limit)

    # Constructors
    eval_function = globals()[parameters.eval_function](parameters.oracle, number_games, seed)
    algorithm = globals()[parameters.search_algorithm](folder_name, parameters.file_name, seed)

    # LOAD INITIAL POLICY
    if parameters.init_program is not None:
        parameters.init_program = pickle.load(open("../LunarLander/binary_programs/" + parameters.init_program, "rb"))

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
        OPERATIONS = [AssignAction,
                          Ite,
                          Lt,
                          Num,
                          Observation,
                          Addition,
                          Multiplication]

        # Add ReLU node to DSL and load ReLU programs
        if parameters.relu_augment is not None:
            OPERATIONS.append(ReLU)
            parameters.relu_augment = pickle.load(open(parameters.relu_augment, "rb"))

        algorithm.search(OPERATIONS,
                         [-0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
                         [0, 1, 2, 3, 4, 5, 6, 7],
                         [0, 1, 2, 3],
                         parameters.relu_augment,
                         eval_function,
                         parameters.use_triage,
                         parameters.use_double_programs,
                         float(parameters.initial_temperature),
                         float(parameters.alpha),
                         float(parameters.beta),
                         time_limit,
                         None,
                         parameters.init_program,
                         bool(parameters.bayes_opt))

if __name__ == "__main__":
    main()
