from gymnasium.envs.registration import register
from RLEnvironment import SimEnv

register(
    id="SimEnv-v0",
    entry_point="RLEnvironment.envs:SimEnv",
)
