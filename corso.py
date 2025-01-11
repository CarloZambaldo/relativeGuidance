import os
import gymnasium
from RLEnvironment import SimEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import time

# load the environment
envName = "SymEnv-v0"
env = gymnasium.make(envName)

# test the environment
episodes = 5
for episode in range(1,episodes+1):
    state = env.reset()
    terminated = False
    truncated = False
    score = 0

    # run the episode
    while not terminated:
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        score += reward

    print(f"Episode: {episode}, Score: {score}")
env.close()

## TRAINING ##

# define the directories for logs
log_dir = f"logs/{int(time.time())}/"

if not os.path.exists(log_dir):
	os.makedirs(log_dir)


# create the environment
env = gymnasium.make(envName)       # this line creates the environment
env = DummyVecEnv([lambda: env])    # wrap the environment in a DummyVecEnv object
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir) # create the model (Multi-Layer Perceptron Policy)

# DELETE THE MODEL AND RELOAD IT
models_dir = os.path.join("models","PPO_test")
model.save(models_dir)
# delete
del model
# reload the model
model = PPO.load(models_dir,env=env)

# EVALUATE THE ENVIRONMENT
evaluate_policy(model, env, n_eval_episodes=10, render=False)


## TEST THE ENVIRONMENT ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# using the model to predict the action to take in the environment based on the observation
# in order to optimize the reward
episodes = 5
for episode in range(1,episodes+1):
    obs = env.reset()
    terminated = False
    truncated = False
    score = 0

    # run the episode
    while not terminated and not truncated:
        action, _ = model.predict(obs) # now using the model 
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
    print(f"Episode: {episode}, Score: {score}")

env.close()




## VIEW THE TENSORBOARD LOGS ##

# give a path
training_log_dir = os.path.join(log_dir,"PPO_1")
os.system(f"tensorboard --logdir={training_log_dir} --host localhost --port 6006")




