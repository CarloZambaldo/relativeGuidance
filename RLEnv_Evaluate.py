from SimEnvRL import *
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import gymnasium as gym

# load the environment
env = gym.make("SimEnv-v1")

# reload the model
RLagent = config.RL_config.recall("PhaseID_2-PPO_v6","latest")
model = PPO.load(RLagent.models_dir,env=env)


# EVALUATE THE ENVIRONMENT
evaluate_policy(model, env, n_eval_episodes=10, render=False)
