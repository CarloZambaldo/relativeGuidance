from SimEnvRL import *
from stable_baselines3 import PPO
import gymnasium as gym

## CHANGE HERE ##
trainingType = "TRAIN_NEW_MODEL"
fileName     = "PPO_PhaseID_2_3rd_try"

# create the environment
env = gym.make('SimEnv-v1',options={"phaseID":2, "tspan": np.array([0, 0.025])})
env.reset(seed=None)

match trainingType:
    case "TRAIN_NEW_MODEL":
        # definition of the learning parameters
        RLparam = config.RL_config.get(fileName)
        # create the model
        model = PPO('MlpPolicy', env=env, verbose=1, tensorboard_log=RLparam.log_dir)

    case "CONTINUE_TRAINING_OLD_MODEL":
        # definition of the learning parameters
        RLparam = config.RL_config.recall(fileName,"latest") # recall latest trained model saved under the given model Name
        model = PPO.load(RLparam.models_dir, env=env)

    case _:
        raise Exception("training Type not defined.")

RLparam.viewLogs()


## TRAINING ##
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=RLparam.maxTimeSteps, reset_num_timesteps=False, tb_log_name=RLparam.modelName)
    model.save(RLparam.models_dir)