from SimEnvRL import *
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys

if len(sys.argv) < 3:
    print("Usage: python3 RLEnv_Training.py <phaseID> <'modelName'> <optional:renderingBool>")
else:
    phaseID = int(sys.argv[1])
    modelName = sys.argv[2]
    renderingBool = False
    if len(sys.argv) == 4:
        renderingBool = 0 if (sys.argv[3] == "False" or sys.argv[3] == "0") else 1

## CHANGE HERE ##
trainingType = "TRAIN_NEW_MODEL"             # "TRAIN_NEW_MODEL" or "CONTINUE_TRAINING_OLD_MODEL"
#modelName    = "Agent_P1-PPO-v4.0-achiral"    # name of the model (to store it or to load it)
deviceType   = "cpu"                         # "cuda" or "cpu"
normalisation = True                         # True or False
discountFactor = 0.99                        # discount factor for the reward
#phaseID = 1
if phaseID == 1:
    tspan = np.array([0, 0.04])
else:
    tspan = np.array([0, 0.03])


# to run tensorboard use: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# tensorboard --logdir="AgentModels//" --host localhost --port 6006

# Create vectorized environments
if deviceType == "cuda": # IF USING GPU
    env = make_vec_env('SimEnv-v4', n_envs=20, env_kwargs={"options":{"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool}})
elif deviceType == "cpu": # IF USING CPU
    if not normalisation: # IF USING CPU without vectorized environment
        env = gym.make('SimEnv-v4',options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})
    else: # IF USING CPU with vectorized environment
        env = DummyVecEnv([lambda: gym.make('SimEnv-v4',options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

print("***************************************************************************")
print(f"Training: {modelName} on {deviceType} with normalisation: {normalisation}")
print(f"Phase ID: {phaseID}, tspan: {tspan}, rendering: {renderingBool}")
print(f"Discount Factor: {discountFactor}")
print("***************************************************************************")
# Reset the environment
env.reset()

# switch case to select between training new model and continue training the old one
match trainingType:
    case "TRAIN_NEW_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.get(modelName)
        # create the model
        model = PPO('MlpPolicy', env=env, device=deviceType, verbose=1, gamma = discountFactor, tensorboard_log=RLagent.log_dir) # USING GPU

    case "CONTINUE_TRAINING_OLD_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.recall(modelName,"latest") # recall latest trained model saved under the given model Name
        model = PPO.load(f"{RLagent.model_dir}/{RLagent.modelNumber}", env=env, device=deviceType, verbose=1, tensorboard_log=RLagent.log_dir)

    case _:
        raise Exception("training Type not defined.")

model.save(RLagent.modelFileNameDir)

## TRAINING ##
print("TRAINING ...")
for iter in range(RLagent.maxIterations):
    model.learn(total_timesteps=RLagent.maxTimeSteps, reset_num_timesteps=True, tb_log_name=RLagent.modelName)
    model.save(RLagent.modelFileNameDir)
    print(f"> TRAINING ITERATION {iter} COMPLETED")
print("STOPPED TRAINING")