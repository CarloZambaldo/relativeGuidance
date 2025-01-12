from SimEnvRL import config
import SimEnvRL
from stable_baselines3 import PPO
import gymnasium as gym


# gym.pprint_registry()

# definition of the learning parameters
trainingType = "CONTINUE_TRAINING_OLD_MODEL"
RLparam = config.RL_config.get("PPO_1_restricted_test")

# create the environment
env = gym.make('SimEnv-v0')
env.reset(seed=None)

match trainingType:
    case "TRAIN_NEW_MODEL":
        # create the model
        model = PPO('MlpPolicy', env=env, verbose=1, tensorboard_log=RLparam.log_dir)

    case "CONTINUE_TRAINING_OLD_MODEL":
        model = PPO.load(RLparam.models_dir, env=env)

    case _:
        raise Exception("training Type not defined.")

## TRAINING ##
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=RLparam.maxTimeSteps, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(RLparam.models_dir)