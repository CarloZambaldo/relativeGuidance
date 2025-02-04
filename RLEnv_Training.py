from SimEnvRL import *
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime
import sys


# to run tensorboard use: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# tensorboard --logdir="AgentModels//" --host localhost --port 6006

## SYSTEM INPUT PARAMETERS ##
if len(sys.argv) < 4:
    phaseID = 2
    modelName = "TEST_AGENT"
    taip = "new"
    renderingBool = True
    print("Usage: python3 RLEnv_Training.py <phaseID> <'modelName'> <'new'/'old'> <optional:renderingBool>")
else:
    phaseID = int(sys.argv[1])
    modelName = sys.argv[2]
    renderingBool = False
    taip = sys.argv[3]
    if len(sys.argv) == 5:
        renderingBool = 0 if (sys.argv[4] == "False" or sys.argv[4] == "0") else 1

## TRAINING MODES ##
if taip == "new":
    trainingType = "TRAIN_NEW_MODEL"             # "TRAIN_NEW_MODEL" or "CONTINUE_TRAINING_OLD_MODEL"
elif taip == "old":
    trainingType = "CONTINUE_TRAINING_OLD_MODEL"
else:
    raise Exception("training Type not defined. Please use 'new' or 'old'.")
#modelName    = "Agent_P1-PPO-v4.0-achiral"    # name of the model (to store it or to load it)
deviceType   = "cpu"                           # "cuda" or "cpu"

#phaseID = 1
if phaseID == 1:
    tspan = np.array([0, 0.06])
else:
    tspan = np.array([0, 0.028745]) # about 3 hours of simulation


## TRAINING PARAMETERS ##
def lr_schedule(progress_remaining):
    return 1e-4 #* progress_remaining    # Decreases as training progresses
norm_reward = False 
norm_obs = False
discountFactor = 0.99       # discount factor for the reward
ent_coef = 0.0001           # entropy coefficient
n_steps = 5000              # consider different trajectories
batch_size = 200            # divisor of n_steps for efficiency recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
n_epochs = 10               # every value is used n times for training


# Create environment (depending on the device and normalisation)
if deviceType == "cpu": # IF USING CPU
    if (norm_reward or norm_obs): # IF USING CPU with vectorized environment
        env = DummyVecEnv([lambda: gym.make('SimEnv-v4.8',options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})])
        # normalize the environment
        env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    else: # IF USING CPU without vectorized environment
        env = gym.make('SimEnv-v4.8', options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})

elif deviceType == "cuda": # IF USING GPU
    env = make_vec_env('SimEnv-v4.8', n_envs=20, env_kwargs={"options":{"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool}})
    raise Exception("GPU not supported on achiral.")


print("\n***************************************************************************")
print("-- AGENT TRAINING PARAMETERS --")
if trainingType == "TRAIN_NEW_MODEL":
    print(f"Training: {modelName} (new) on {deviceType}")
    print(f"Phase ID:\t{phaseID}\ntspan:   \t{tspan}\nrendering:\t{renderingBool}")
    print(f"norm_reward: {norm_reward}; norm_obs = {norm_obs}")
    print(f"gamma:     \t{discountFactor}\nent_coef:\t{ent_coef}\nlearning_rate:\tlinear from {lr_schedule(1)} to {lr_schedule(0)}")
    print(f"n_steps:\t{n_steps}\nbatch_size:\t{batch_size}\nn_epochs:\t{n_epochs}")
else:
    print(f"Training: {modelName} (continue) on {deviceType}.\nNOTE: training parameters are not used.")
print("***************************************************************************\n")

# Reset the environment
print("RESETTING THE ENVIRONMENT...")
env.reset()

# switch case to select between training new model and continue training the old one
match trainingType:
    case "TRAIN_NEW_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.get(modelName)
        # create the model
        model = PPO('MlpPolicy', env=env,
                    learning_rate=lr_schedule,
                    ent_coef=ent_coef,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    device=deviceType, verbose=1, gamma = discountFactor, tensorboard_log=RLagent.log_dir) # USING GPU
    case "CONTINUE_TRAINING_OLD_MODEL":
        # definition of the learning parameters
        RLagent = config.RL_config.recall(modelName,"latest") # recall latest trained model saved under the given model Name
        model = PPO.load(f"{RLagent.model_dir}/{RLagent.modelNumber}", env=env, device=deviceType, verbose=1, tensorboard_log=RLagent.log_dir)
    case _:
        raise Exception("training Type not defined.")

# save the starting model to ensure it will save
model.save(RLagent.modelFileNameDir)

## TRAINING ##
print("TRAINING ...")
model.learn(total_timesteps=RLagent.maxTimeSteps, reset_num_timesteps=True, tb_log_name=RLagent.modelName)
model.save(RLagent.modelFileNameDir)
print(f"FINISHED TRAINING: {datetime.now().strftime('%Y/%m/%d AT %H:%M')}")



# output to recall which model was trained in that window
if trainingType == "TRAIN_NEW_MODEL":
    print("\n***************************************************************************")
    print("-- TRAINED (NEW) AGENT USING FOLLOWING PARAMETERS --")
    print(f"Trained: {modelName} on {deviceType}")
    print(f"norm_reward: {norm_reward}; norm_obs = {norm_obs}")
    print(f"Phase ID:\t{phaseID}\ntspan:   \t{tspan}\nrendering:\t{renderingBool}")
    print(f"gamma:     \t{discountFactor}\nent_coef:\t{ent_coef}\nlearning_rate:\tlinear from {lr_schedule(1)} to {lr_schedule(0)}")
    print(f"n_steps:\t{n_steps}\nbatch_size:\t{batch_size}\nn_epochs:\t{n_epochs}")
    print("***************************************************************************\n")