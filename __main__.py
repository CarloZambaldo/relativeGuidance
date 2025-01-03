from imports import *

param, initialValue = config.env_config.get()
#RLparam = config.agent_config.get()

# loading RL environment
env = RLE.RLEnvironment(param, initialValue)

## check the environment ##
episodes = 50

for episode in range(episodes):
	state = env.reset()
	done = False
	score = 0

	while not done:
		action = env.action_space.sample()
		n_state, reward, done, info = env.step(action)
		score += reward
	print(f'Episode: {episode}, Score: {score}')

## TRAINING ##
from stable_baselines3 import PPO
import os
from snakeenv import SnekEnv
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = RLenv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")

## DEPLOYMENT ##


## PERFORMANCE EVALUATION ##


## PLOTTING ##

