import pickle
from evaluation import *

policy = pickle.load(open("../LunarLander/binary_programs/sa-1-cpus-program_test_Imitation.pkl", "rb"))
print(policy.to_string())

avg_reward = Environment(25).eval_render(policy)[0]
print("\nAverage reward", avg_reward)
