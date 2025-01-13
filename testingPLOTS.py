import pickle
from SimEnvRL import *

fileNumber = "1736704301"

with open(f"Simulations/{fileNumber}.pkl", "rb") as file:
    env = pickle.load(file)
    
printSummary(env)
plotty(env)
