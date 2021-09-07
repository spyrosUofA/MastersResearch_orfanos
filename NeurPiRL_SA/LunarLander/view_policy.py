import pickle
from evaluation import *
import pandas as pd


if True:
    trajs = pd.read_csv("../LunarLander/trajectory.csv")
    observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]', 'o[4]', 'o[5]', 'o[6]', 'o[7]']].to_numpy()
    actions = trajs['a'].to_numpy()
    scorer = Imitation(25)
    scorer.add_trajectory(observations, actions)

    p1 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_InitialProgram.pkl", "rb"))
    print("Score 1: ",  scorer.evaluate(p1)[0])

    p2 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))
    print("Score 2: ",  scorer.evaluate(p2)[0])

    p3 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test5.pkl", "rb"))
    print("Score 3: ", scorer.evaluate(p3)[0])
    print()
    #exit()


if True:
    p1 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_InitialProgram.pkl", "rb"))
    print("Avg. Reward 1: ",  Environment(25).evaluate(p1)[0])

    p2 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))
    print("Avg. Reward 2: ", Environment(25).evaluate(p2)[0])

    p3 = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test5.pkl", "rb"))
    print("Avg. Reward 3: ", Environment(25).evaluate(p3)[0])
    print()
    exit()



policy = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))
print(policy.to_string())
avg_reward = Environment(25).eval_render(policy)[0]
print("\nAverage reward", avg_reward)