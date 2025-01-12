from SimEnvRL import config
import SimEnvRL
from stable_baselines3 import PPO
import gymnasium as gym


# gym.pprint_registry()

# definition of the learning parameters
RLparam = config.RL_config.get("PPO_1_restricted_test")

# create the environment
env = gym.make('SimEnv-v0')
env.reset(seed=None)

# create the model
model = PPO('MultiInputPolicy', env=env, verbose=1, tensorboard_log=RLparam.log_dir)

iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=RLparam.maxTimeSteps, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(RLparam.models_dir)