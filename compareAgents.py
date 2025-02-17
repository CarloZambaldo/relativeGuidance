import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from SimEnvRL import *
from stable_baselines3 import PPO

env1 = gym.make("SimEnv-v4.8", options={"phaseID":1,"tspan":np.array([0, 0.045]),"renderingBool":False})
env2 = gym.make("SimEnv-v4.8", options={"phaseID":2,"tspan":np.array([0, 0.033]),"renderingBool":False})
model1 = PPO.load(r"AgentModels/Agent_P1-v11.3.p1-multi-phase1-SEMIDEF/model/1739005750.zip", env=env1, device="cpu")
model2 = PPO.load(r"AgentModels/Agent_P2-v11.5-multi-SEMIDEF/model/1739048517.zip", env=env2, device="cpu")

import torch

# Extract the parameters of the policy network
weights_network1 = {name: param.data.clone() for name, param in model1.policy.named_parameters()}
weights_network2 = {name: param.data.clone() for name, param in model2.policy.named_parameters()}

# Compare biases separately
biases_network1 = {name: param for name, param in weights_network1.items() if "bias" in name}
biases_network2 = {name: param for name, param in weights_network2.items() if "bias" in name}

# Print bias values for comparison
for key in biases_network1:
    print(f"Layer: {key}")
    print("Bias Model 1:", biases_network1[key].numpy())
    print("Bias Model 2:", biases_network2[key].numpy())
    print("Difference:", (biases_network1[key] - biases_network2[key]).numpy())
    print("-" * 40)


