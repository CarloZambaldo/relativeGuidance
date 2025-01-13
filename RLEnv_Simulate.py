from SimEnvRL import *
from stable_baselines3 import PPO
import pickle
import time 
import os

env = gym.make("SimEnv-v0")  # this line creates the environment

# load the model
RLparam = config.RL_config.recall("PPO_PhaseID_2_RestrictedTest","latest")
model = PPO.load(RLparam.models_dir, env=env)

# run the episodes
episodes = 1
for episode in range(episodes):
    # reset the episode
    print(f"\n## RUN {episode+1} out of {episodes} ##\n")
    terminated = False
    truncated = False
    obs, info = env.reset()

    # run each episode:
    while (not terminated and not truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    print(f"Episode: {episode}, Total Reward: {reward}")
    print("############\n\n")

    # Save the environment
    new_dir = "Simulations"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    file_path = os.path.join(new_dir, f"{int(time.time())}.pkl")

    with open(file_path, 'wb') as f:
        pickle.dump(env, f)

    if terminated:
        indicezeri = (env.fullStateHistory[:, 6:12] == 0)
        env.fullStateHistory[indicezeri, 0:6] = 0

        indiceValori = not (env.fullStateHistory[:, 0] == 0)
        env.fullStateHistory = env.fullStateHistory[indiceValori, :]
        env.controlActionHistory_L = env.controlActionHistory_L[indiceValori, :]
        env.timeHistory = env.timeHistory[indiceValori]

    printSummary(env)
    plotty(env)
    input("Press enter to continue...")
env.close()

