from SimEnvRL import *
from stable_baselines3 import PPO
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import gymnasium as gym

# Parametri
phaseID = 2
tspan = np.array([0, 0.02])
agentName = "Agent_P2-PPO-v5-CUDA"
renderingBool = True

# Creazione dell'ambiente
env = gym.make("SimEnv-v2", options={"phaseID": phaseID, "tspan": tspan})

# Caricamento del modello RL
RLagent = config.RL_config.recall(agentName, "latest")
model = PPO.load(f"{RLagent.model_dir}\\{RLagent.modelNumber}", env=env, device="cpu")

# Funzione per il profiling
def run_episode():
    terminated = False
    truncated = False
    obs, info = env.reset()

    while not (terminated or truncated):
        with record_function("model_prediction"):
            action, _ = model.predict(obs)  # Predict dell'agente
        with record_function("environment_step"):
            obs, reward, terminated, truncated, info = env.step(action)  # Step dell'ambiente
        if renderingBool:
            with record_function("rendering"):
                env.render()

# Profiling con PyTorch Profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as prof:
    run_episode()

# Risultati del Profiling
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
