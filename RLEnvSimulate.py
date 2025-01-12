from SimEnvRL import *
from stable_baselines3 import PPO

env = gym.make(SimEnv)  # this line creates the environment

# load the model
models_dir = os.path.join("AgentModels","PPO_0")
model = PPO.load(models_dir,env=env)

episodes = 1

for episode in range(episodes):
    print(f"## RUN {episode+1} out of {episodes} ##\n")
    terminated = False
    truncated = False
    obs, info = env.reset(None)
    while (not terminated and not truncated):
        action, _ = model.predict(obs)
        print("Agent Action: ",action)
        obs, reward, terminated, truncated, info = env.step(action)
        print('Reward: ',reward)
        if info["timeNow"] > info["param"].tspan[-1]:
            truncated = True
    print(f"Episode: {episode}, Total Reward: {reward}")

    print("############\n\n")
    printSummary(env)
    plotty(env)
    input("Press enter to continue...")
env.close()

# TENSORBOARD LOGS
log_dir = os.path.join("AgentModels","logs")
training_log_dir = os.path.join(log_dir,"PPO_0")
os.system(f"tensorboard --logdir={training_log_dir} --host localhost --port 6006")


