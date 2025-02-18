import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from SimEnvRL import *
from stable_baselines3 import PPO
import torch

env1 = gym.make("SimEnv-v4.8", options={"phaseID":1,"tspan":np.array([0, 0.045]),"renderingBool":False})
env2 = gym.make("SimEnv-v4.8", options={"phaseID":2,"tspan":np.array([0, 0.033]),"renderingBool":False})
model1 = PPO.load(r"./AgentModels/Agent_P1-v11.3.p1-multi-phase1-SEMIDEF/model/1739005750.zip", env=env1, device="cpu")
model2 = PPO.load(r"./AgentModels/Agent_P2-v11.5-multi-SEMIDEF/model/1739048517.zip", env=env2, device="cpu")



# Extract weights for each layer and compute similarity
def get_layerwise_similarity(model1, model2):
    similarities = {"policy": [], "value": []}
    layer_sizes = {"policy": [], "value": []}
    layer_names = {"policy": [], "value": []}
    
    for (name1, param1), (name2, param2) in zip(model1.policy.named_parameters(), model2.policy.named_parameters()):
        if 'weight' in name1:  # Only consider weight matrices
            param1_np = param1.data.cpu().numpy()
            param2_np = param2.data.cpu().numpy()

            # Compute cosine similarity per neuron (row-wise)
            layer_sim = [cosine_similarity([p1], [p2])[0, 0] for p1, p2 in zip(param1_np, param2_np)]

            # Classify layer type
            if "policy_net" in name1 or "action_net" in name1:
                similarities["policy"].append(layer_sim)
                layer_sizes["policy"].append(len(layer_sim))
                layer_names["policy"].append(name1)
            elif "value_net" in name1:
                similarities["value"].append(layer_sim)
                layer_sizes["value"].append(len(layer_sim))
                layer_names["value"].append(name1)

    return similarities, layer_sizes, layer_names

# Compute similarities
similarities, layer_sizes, layer_names = get_layerwise_similarity(model1, model2)

# Create figure with two columns
fig, axes = plt.subplots(max(len(similarities["policy"]), len(similarities["value"])), 2, figsize=(12, 10))


# Define custom colormap with enhanced colors
from matplotlib.colors import LinearSegmentedColormap
colors = ["#3361d4", "#FFFFFF", "#00AA00"]  # Dimmed blue, white, vivid green
positions = [0, 0.5, 1]  # -1 -> blue, 0 -> white, 1 -> green
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))

# Plot policy network (left column)
for i, (sim, name) in enumerate(zip(similarities["policy"], layer_names["policy"])):
    sim_array = np.array(sim).reshape(-1, 1)
    ax = axes[i, 0] if len(similarities["policy"]) > 1 else axes[0]
    sns.heatmap(sim_array, cmap=custom_cmap, vmin=-1, vmax=1, annot=False, cbar=True, ax=ax)
    ax.set_title(f'Policy: {name}')
    ax.set_ylabel("Neuron Index")
    ax.set_xticks([])
    if len(sim_array) > 10:
        ax.set_yticks([0,8,16,32,48,56,64],[0,8,16,32,48,56,64])

# Plot value network (right column)
for i, (sim, name) in enumerate(zip(similarities["value"], layer_names["value"])):
    sim_array = np.array(sim).reshape(-1, 1)
    ax = axes[i, 1] if len(similarities["value"]) > 1 else axes[1]
    sns.heatmap(sim_array, cmap=custom_cmap, vmin=-1, vmax=1, annot=False, cbar=True, ax=ax)
    ax.set_title(f'Value: {name}')
    ax.set_ylabel("Neuron Index")
    ax.set_xticks([])
    if len(sim_array) > 10:
        ax.set_yticks([0,8,16,32,48,56,64],[0,8,16,32,48,56,64])

# Set column titles
axes[0, 0].set_title("Policy Network")
axes[0, 1].set_title("Value Network")



###########################################################
# Function to extract activations for a given input
def get_activations(model, input_tensor):
    activations = []
    layer_names = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    hooks = []
    for i, layer in enumerate(model.policy.mlp_extractor.policy_net):
        hooks.append(layer.register_forward_hook(hook_fn))
        layer_names.append(f"policy_net.{i}")  # Naming based on structure

    _ = model.policy(input_tensor)  # Forward pass to get activations

    for hook in hooks:
        hook.remove()

    return np.array(activations), layer_names

# Function to compute mean activations over multiple random initializations
def compute_mean_activations(model, input_dim, num_runs=10, seed_base=42):
    activations_all = []

    for i in range(num_runs):
        seed = seed_base + i
        torch.manual_seed(seed)  # Set random seed
        input_tensor = torch.randn(1, input_dim).to(model.device)  # Generate random input
        activations, layer_names = get_activations(model, input_tensor)
        activations_all.append(activations)

    # Compute mean activations across runs
    mean_activations = np.mean(activations_all, axis=0)

    return mean_activations, layer_names

# Set parameters
num_runs = 100  # Number of different initializations
input_dim = 10  # Input dimension for the model

# Compute averaged activations for both models in parallel
mean_act1, layers1 = compute_mean_activations(model1, input_dim, num_runs)
mean_act2, layers2 = compute_mean_activations(model2, input_dim, num_runs)

# Visualization
fig, axes = plt.subplots(len(layers1), 2, figsize=(12, len(layers1) * 2))

# Plot Mean Activations for Both Models
for i, (act1, act2, layer_name) in enumerate(zip(mean_act1, mean_act2, layers1)):
    sns.heatmap(act1.reshape(-1, 1), cmap='Blues', ax=axes[i, 0], cbar=False)
    axes[i, 0].set_title(f'Mean {layer_name} Activations - Model 1')
    axes[i, 0].set_xticks([])

    sns.heatmap(act2.reshape(-1, 1), cmap='Greens', ax=axes[i, 1], cbar=False)
    axes[i, 1].set_title(f'Mean {layer_name} Activations - Model 2')
    axes[i, 1].set_xticks([])



##########################

# Adjust layout for better readability
plt.tight_layout()
plt.show()

# import torch
# 
# # Extract the parameters of the policy network
# weights_network1 = {name: param.data.clone() for name, param in model1.policy.named_parameters()}
# weights_network2 = {name: param.data.clone() for name, param in model2.policy.named_parameters()}
# 
# # Compare biases separately
# biases_network1 = {name: param for name, param in weights_network1.items() if "bias" in name}
# biases_network2 = {name: param for name, param in weights_network2.items() if "bias" in name}
# 
# # Print bias values for comparison
# for key in biases_network1:
#     print(f"Layer: {key}")
#     print("Bias Model 1:", biases_network1[key].numpy())
#     print("Bias Model 2:", biases_network2[key].numpy())
#     print("Difference:", (biases_network1[key] - biases_network2[key]).numpy())
#     print("-" * 40)
# 
# 
# # Extract weights from PPO policy networks
# def get_flattened_weights(model):
#     weights = [param.data.cpu().numpy().flatten() for name, param in model.policy.named_parameters()]
#     return np.concatenate(weights)
# 
# # Get flattened weights for both models
# weights1_flat = get_flattened_weights(model1)
# weights2_flat = get_flattened_weights(model2)
# 
# # Compute cosine similarity
# similarity = cosine_similarity([weights1_flat], [weights2_flat])
# 
# # Plot heatmap
# plt.figure(figsize=(6, 5))
# sns.heatmap(similarity, cmap='RdYlGn', annot=True, cbar=True)
# plt.title('Cosine Similarity Between Networks')
# plt.xlabel('Network 2')
# plt.ylabel('Network 1')
# plt.show()