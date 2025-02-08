from SimEnvRL import *
from stable_baselines3 import PPO
import torch
import gymnasium as gym
import numpy as np
import pickle
from stable_baselines3.common.vec_env import VecNormalize
import matplotlib.pyplot as plt

def printNeuralNetwork(model):
    # Accedere alla policy (rete neurale interna del PPO)
    policy_net = model.policy  # Questo è il modello PyTorch effettivo

    # Funzione per stampare i parametri della policy network con controllo su std dev
    def print_policy_parameters(policy_net):
        print("\nPolicy Network Parameters:\n" + "-"*50)
        for name, paramet in policy_net.named_parameters():
            if paramet.requires_grad:
                print(f"Layer: {name}")
                print(f" - Shape: {paramet.shape}")
                print(f" - Mean: {paramet.data.mean().item():.6f}")

                # Controllo per evitare il calcolo della std su tensori con un solo elemento
                if paramet.numel() > 1:
                    print(f" - Std Dev: {paramet.data.std().item():.6f}")
                else:
                    print(" - Std Dev: N/A (single element)")

                print(f" - Min: {paramet.data.min().item():.6f}")
                print(f" - Max: {paramet.data.max().item():.6f}")
                print("-"*50)

    # Stampa i parametri della policy network
    print_policy_parameters(policy_net)

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


def printNormalization(RLagent):
    with open(f"{RLagent.model_dir}/vec_normalize.pkl", "rb") as file:
        vec_norm = pickle.load(file)

    # Print normalization statistics
    print("*"*30)
    print("Observation Mean:", vec_norm.obs_rms.mean)
    print("Observation Variance:", vec_norm.obs_rms.var)
    print("Reward Mean:", vec_norm.ret_rms.mean)
    print("Reward Variance:", vec_norm.ret_rms.var)
    print("Clip Range:", vec_norm.clip_obs, vec_norm.clip_reward)
    print("*"*30)