from stable_baselines3.common.env_checker import check_env
from RLEnvironment import SimEnv
import time
from imports import *
env = SimEnv()

# first check of the environment definition
check_env(env)
print("ENVIRONMENT CHECKED!\n")

# doublecheck the environment definition
episodes = 1

for episode in range(episodes):
	print(f"## RUN {episode+1} out of {episodes} ##\n")
	terminated = False
	truncated = False
	obs, info = env.reset(seed=None,options={"phaseID":1})
	while (not terminated and not truncated):
		random_action = 0#env.action_space.sample()
		print("Agent Action: ",random_action)
		obs, reward, terminated, truncated, info = env.step(random_action)
		print('Reward: ',reward)

		if info["timeNow"] > info["param"].tspan[-1]:
			truncated = True

	print("############\n\n")
	printSummary(env)
	#plotty(env)
	#input("Press enter to continue...")
		
import pickle
with open("savedEnvironmentTEST.pkl", "wb") as file:
	pickle.dump(env, file)
	print("SAVED env INTO A FILE.")

## TRAINING ##
