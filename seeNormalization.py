import pickle
from stable_baselines3.common.vec_env import VecNormalize
from SimEnvRL import *

modelName = "TEST"
# Load the saved VecNormalize object
RLagent = config.RL_config.get(modelName)
with open(f"{RLagent.model_dir}/vec_normalize.pkl", "rb") as file:
    vec_norm = pickle.load(file)


# Print normalization statistics
print("*"*30)
print("Observation Mean:", vec_norm.obs_rms.mean)
print("Observation Variance:", vec_norm.obs_rms.var)
print("Reward Mean:", vec_norm.ret_rms.mean)
print("Reward Variance:", vec_norm.ret_rms.var)
print("Clip Range:", vec_norm.clip_obs, vec_norm.clip_reward)
print("*"*30)