from SimEnvRL import *
from stable_baselines3 import PPO
import torch
import gymnasium as gym
import numpy as np

# Caricamento dell'agente
agentName = "Agent_P2-PPO-v5.0-s"
phaseID = 2
if phaseID == 1:
	tspan = np.array([0, 0.04]) # ca 4 hours
elif phaseID == 2:
	tspan = np.array([0, 0.033]) # ca 3.3 hours

env = gym.make("SimEnv-v4.8", options={"phaseID": phaseID, "tspan": tspan, "renderingBool": False})
RLagent = config.RL_config.recall(agentName, "latest")
model = PPO.load(f"{RLagent.model_dir}/{RLagent.modelNumber}", env=env, device='cpu')

# Accedere alla policy (rete neurale interna del PPO)
policy_net = model.policy  # Questo è il modello PyTorch effettivo

# Funzione per stampare i parametri della policy network con controllo su std dev
def print_policy_parameters(policy_net):
	print("\nPolicy Network Parameters:\n" + "-"*50)
	for name, param in policy_net.named_parameters():
		if param.requires_grad:
			print(f"Layer: {name}")
			print(f" - Shape: {param.shape}")
			print(f" - Mean: {param.data.mean().item():.6f}")

			# Controllo per evitare il calcolo della std su tensori con un solo elemento
			if param.numel() > 1:
				print(f" - Std Dev: {param.data.std().item():.6f}")
			else:
				print(" - Std Dev: N/A (single element)")

			print(f" - Min: {param.data.min().item():.6f}")
			print(f" - Max: {param.data.max().item():.6f}")
			print("-"*50)

# Stampa i parametri della policy network
print_policy_parameters(policy_net)



import matplotlib.pyplot as plt

def plot_weights_histogram(policy_net):
	for name, param in policy_net.named_parameters():
		if param.requires_grad:
			plt.figure()
			plt.hist(param.data.cpu().numpy().flatten(), bins=50)
			plt.title(f"Histogram of {name}")
			plt.xlabel("Weight value")
			plt.ylabel("Frequency")


plot_weights_histogram(policy_net)

plt.show()