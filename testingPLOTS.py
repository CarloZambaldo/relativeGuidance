import pickle
from imports import *

with open("savedEnvironmentTEST.pkl", "rb") as file:
    env = pickle.load(file)
    
plotty(env) 