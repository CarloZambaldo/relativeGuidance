from gymnasium.envs.registration import register
from RLEnvironment import *

register(
    id="myEnv/RLenv-v0",
    entry_point="RLenv.envs:simulationEnvironment",
)
