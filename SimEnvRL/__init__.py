# standard libraries imports
import os
import time

# RL imports
import gymnasium as gym
from gymnasium.envs.registration import register

# scientific imports
import numpy as np
import scipy

# custom functions imports
from SimEnvRL import config              # this allows to call config inside a script without importing it
from SimEnvRL.UserDataDisplay.printSummary import printSummary
from SimEnvRL.UserDataDisplay.plots import plotty



## REGISTER THE ENVIRONMENT ##
register(
    id="SimEnv-v1",
    entry_point="SimEnvRL.envs.RLEnvironment:SimEnv",
)

