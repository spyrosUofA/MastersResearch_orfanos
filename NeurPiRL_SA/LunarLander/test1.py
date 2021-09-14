import pandas as pd

trajs = pd.read_csv("../LunarLander/trajectory.csv")
observations = trajs[['o[0]', 'o[1]', 'o[2]', 'o[3]', 'o[4]', 'o[5]', 'o[6]', 'o[7]']].to_numpy()
actions = trajs['a'].to_numpy()

print(observations)