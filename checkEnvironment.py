from stable_baselines3.common.env_checker import check_env
from RLEnvironment import SimEnv


envOptions = { "phaseID":1 }
env = SimEnv()

# first check of the environment definition
# check_env(env)
# print("ENVIRONMENT CHECKED!\n")

# doublecheck the environment definition
episodes = 10

for _ in range(episodes):
	terminated = False
	obs = env.reset(None,envOptions)
	while not terminated:
		random_action = env.action_space.sample()
		print("action: ",random_action)
		obs, reward, terminated, truncated, info = env.step(random_action)
		print('reward: ',reward)
		

## TRAINING ##
