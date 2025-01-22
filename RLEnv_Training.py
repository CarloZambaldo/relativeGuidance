from SimEnvRL import *
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

## CHANGE HERE ##
trainingType = "TRAIN_NEW_MODEL" # "TRAIN_NEW_MODEL" or "CONTINUE_TRAINING_OLD_MODEL"
modelName    = "Agent_P2-PPO-v7"             # name of the model (to store it or to load it)
deviceType   = "cpu"                         # "cuda" or "cpu"

phaseID = 2
tspan = np.array([0, 0.03])
# to run tensorboard use:
# tensorboard --logdir="AgentModels//" --host localhost --port 6006

# Create vectorized environments
if deviceType == "cuda": # USING GPU
    env = make_vec_env('SimEnv-v3', n_envs=20, env_kwargs={"options":{"phaseID": phaseID, "tspan": tspan}})
elif deviceType == "cpu": # USING CPU
    env = gym.make('SimEnv-v3',options={"phaseID": phaseID, "tspan": tspan})

# Reset the environment
env.reset()

# switch case to select between training new model and continue training the old one
match trainingType:
    case "TRAIN_NEW_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.get(modelName)
        # create the model
        model = PPO('MlpPolicy', env=env, device=deviceType, verbose=1, gamma = 0.991, tensorboard_log=RLagent.log_dir) # USING GPU

    case "CONTINUE_TRAINING_OLD_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.recall(modelName,"latest") # recall latest trained model saved under the given model Name
        model = PPO.load(f"{RLagent.model_dir}\\{RLagent.modelNumber}", env=env, device=deviceType, verbose=1, tensorboard_log=RLagent.log_dir)

    case _:
        raise Exception("training Type not defined.")

model.save(RLagent.modelFileNameDir)
## TRAINING ##
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=RLagent.maxTimeSteps, reset_num_timesteps=False, tb_log_name=RLagent.modelName)
    model.save(RLagent.modelFileNameDir)