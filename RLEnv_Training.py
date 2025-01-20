from SimEnvRL import *
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

## CHANGE HERE ##
trainingType = "CONTINUE_TRAINING_OLD_MODEL"
modelName     = "Agent_P2-PPO-v5-CUDA"

# to run tensorboard use:
# tensorboard --logdir="AgentModels//" --host localhost --port 6006

# Create vectorized environments
#env = make_vec_env('SimEnv-v2', n_envs=20, env_kwargs={"options":{"phaseID": 2, "tspan": np.array([0, 0.025])}})
env = gym.make('SimEnv-v2',options={"phaseID":2, "tspan": np.array([0, 0.025])})
env.reset()

match trainingType:
    case "TRAIN_NEW_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.get(modelName)
        # create the model
        model = PPO('MlpPolicy', env=env, device="cuda", verbose=1, gamma = 0.991, tensorboard_log=RLagent.log_dir)

    case "CONTINUE_TRAINING_OLD_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.recall(modelName,"latest") # recall latest trained model saved under the given model Name
        model = PPO.load(f"{RLagent.model_dir}\\{RLagent.modelNumber}", env=env, device="cuda", verbose=1, tensorboard_log=RLagent.log_dir)

    case _:
        raise Exception("training Type not defined.")


## TRAINING ##
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=RLagent.maxTimeSteps, reset_num_timesteps=True, tb_log_name=RLagent.modelName)
    model.save(RLagent.modelFileNameDir)