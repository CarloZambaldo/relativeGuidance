from stable_baselines3.common.env_checker import check_env
from RLEnvironment import SimEnv


envOptions = { "phaseID":1 }
env = SimEnv()

# first check of the environment definition
check_env(env)

# doublecheck the environment definition
episodes = 50

for _ in range(episodes):
	done = False
	obs = env.reset(any,1)
	while not terminated:
		random_action = env.action_space.sample()
		print("action: ",random_action)
		obs, reward, terminated, truncated, info = env.step(random_action)
		print('reward: ',reward)
		

## TRAINING ##
