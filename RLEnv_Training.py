from SimEnvRL import *
from stable_baselines3 import PPO
import gymnasium as gym

## CHANGE HERE ##
trainingType = "CONTINUE_TRAINING_OLD_MODEL"
modelName     = "PhaseID_2-PPO_4"

# create the environment
env = gym.make('SimEnv-v1',options={"phaseID":2, "tspan": np.array([0, 0.025])})
env.reset(seed=None)

match trainingType:
    case "TRAIN_NEW_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.get(modelName)
        # create the model
        model = PPO('MlpPolicy', env=env, verbose=1, tensorboard_log=RLagent.log_dir)

    case "CONTINUE_TRAINING_OLD_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.recall(modelName,"latest") # recall latest trained model saved under the given model Name
        model = PPO.load(RLagent.model_dir, env=env, verbose=1, tensorboard_log=RLagent.log_dir)

    case _:
        raise Exception("training Type not defined.")


## TRAINING ##
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=RLagent.maxTimeSteps, reset_num_timesteps=True, tb_log_name=RLagent.modelName)
    model.save(RLagent.model_dir)