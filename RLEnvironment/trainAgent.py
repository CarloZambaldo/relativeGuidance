#
#
#
# Script for training the Agent using PPO
#
from RLEnvironment import RLenv
from stable_baselines3 import PPO

# Crea l'ambiente
env = RLenv()

# Train PPO Agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluation of perfomances
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break