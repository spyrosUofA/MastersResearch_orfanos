import pickle
from evaluation import *

policy = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))
policy = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_InitialProgram.pkl", "rb"))
print(policy.to_string())

NB_EPISODES = 25
avg_reward = Evaluate(NB_EPISODES).eval_render(policy)
print("\nAverage reward", avg_reward)