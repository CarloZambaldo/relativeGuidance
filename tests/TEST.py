import config
#from imports import *
import numpy as np
from numpy import linspace
import os

os.system('cls' if os.name == 'nt' else 'clear')

param, initialValue = config.env_config.get()

print(param)
print("\n\n")

OBoptimalTrajectory = {
    "time": linspace(0, 1, 100),
    "state": np.array([linspace(0, 1, 100),linspace(0, 1, 100),linspace(0, 1, 100),linspace(0, 1, 100),linspace(0, 1, 100),linspace(0, 1, 100)])
}

closestOptimalState = np.array([
    np.interp(.2, OBoptimalTrajectory['time'], OBoptimalTrajectory['state'][i])
    for i in range(6)
])

print(closestOptimalState)

print("\n\n")