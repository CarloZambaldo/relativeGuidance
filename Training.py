from SimEnvRL import *
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, SubprocVecEnv
from datetime import datetime
import torch
import argparse
from torch.utils.data import DataLoader
import os

# to run tensorboard use: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# tensorboard --logdir="AgentModels//" --host localhost --port 6006

if __name__ == "__main__":
    # Create Argument Parser
    parser = argparse.ArgumentParser(description="Reinforcement Learning Experiment")

    # Add argument for reward normalization
    parser.add_argument("-p", "--phase", type=int, default="None", help="Mission Phase")
    parser.add_argument("-m", "--model", type=str, default="None", help="Mission Phase")
    parser.add_argument("-r", "--render", type=str, default="False", help="Rendering bool")
    parser.add_argument("-f","--start-from", type=str, default="new", help="Name of the agent from which continue training")
    # Parse arguments
    argspar = parser.parse_args()

    ## SYSTEM INPUT PARAMETERS ##
    # if len(sys.argv) < 3:
    #     phaseID = 2
    #     modelName = "TEST_AGENT"
    #     taip = argspar.render
    #     renderingBool = True
    #     print("Usage: python3 RLEnv_Training.py --phase <phaseID> --model <'modelName'> --start-from <OldAgentName> --render <renderingBool>")
    # else:
    phaseID = argspar.phase
    modelName = argspar.model
    renderingBool = False if argspar.render == "False" else True

    ## TRAINING MODES ##
    if argspar.start_from == "new":
        trainingType = "TRAIN_NEW_MODEL"             # "TRAIN_NEW_MODEL" or "CONTINUE_TRAINING_OLD_MODEL"
    else:
        trainingType = "CONTINUE_TRAINING_OLD_MODEL"
        modelNameOLD = argspar.start_from
    #modelName    = "Agent_P1-PPO-v4.0-achiral"    # name of the model (to store it or to load it)

    deviceType   = "cpu"                           # "cuda" or "cpu"

    # change training time span depending on the phase
    if phaseID == 1:
        tspan = np.array([0, 0.06])
    else:
        tspan = np.array([0, 0.031]) # np.array([0, 0.028745]) # about 3 hours of simulation

    #get maximum number of threads available
    max_num_threads = torch.get_num_threads()

    ## TRAINING PARAMETERS ##
    def lr_schedule(progress_remaining):
        return (3e-5 - 1e-6) * progress_remaining + 1e-6    # Decreases as training progresses

    if phaseID == 1:
        n_envs          = 15   
        norm_reward     = True 
        norm_obs        = True
        discountFactor  = 0.99    # discount factor for the reward
        ent_coef        = 0.00015  # entropy coefficient
        n_steps         = int(np.ceil(7500/n_envs))    # consider different trajectories
        batch_size      = 250     # divisor of n_steps for efficiency recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
        n_epochs        = 15      # every value is used n times for training
        vf_coef         = 0.5     # value function coefficient
        clip_range      = 0.15    # default: 0.2
        gae_lambda      = 0.95    # default: 0.95
        total_timesteps = 1.5e6   # <<<<<<<<<<<<<<<<<<<<<<<<

    if phaseID == 2:
        n_envs          = 15   
        norm_reward     = True 
        norm_obs        = True
        discountFactor  = 0.99    # discount factor for the reward
        ent_coef        = 0.0002  # entropy coefficient
        n_steps         = int(np.ceil(7500/n_envs))    # consider different trajectories
        batch_size      = 250     # divisor of n_steps for efficiency recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
        n_epochs        = 15      # every value is used n times for training
        vf_coef         = 0.55    # value function coefficient
        clip_range      = 0.15    # default: 0.2
        gae_lambda      = 0.95    # default: 0.95
        total_timesteps = 1.5e6   # <<<<<<<<<<<<<<<<<<<<<<<<

    if n_envs > max_num_threads:
        raise BrokenPipeError("n_envs > max_num_threads")
    
    # definition of the learning parameters
    RLagent = config.RL_config.get(modelName)

    # Create environment (depending on the device and normalisation)
    if deviceType == "cpu": # IF USING CPU
        if (norm_reward or norm_obs): # IF USING CPU with normalized environment
            #env = DummyVecEnv([lambda: gym.make('SimEnv-v5.0',options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})])
            env = SubprocVecEnv([lambda: gym.make('SimEnv-v5.0',options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})
                                  for _ in range(n_envs)])
            env = VecMonitor(env, RLagent.log_dir)  # Logs true episode rewards
            env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward) # normalize the environment
        else: # IF USING CPU without normalized environment
            env =  SubprocVecEnv([lambda: gym.make('SimEnv-v5.0',options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})
                                  for _ in range(n_envs)])
    elif deviceType == "cuda": # IF USING GPU
        env = make_vec_env('SimEnv-v5.0', n_envs=20, env_kwargs={"options":{"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool}})
        raise Exception("GPU not supported on achiral.")


    print("\n***************************************************************************")
    print("-- AGENT TRAINING PARAMETERS --")
    if trainingType == "TRAIN_NEW_MODEL":
        print(f"Training: {modelName} (new) on {deviceType}")
    else:
        print(f"Training: {modelName} (continue) from {modelNameOLD} on {deviceType}.")
    print(f"Using {max_num_threads} threads, running {n_envs} environments in parallel.")
    print(f"Phase ID:\t{phaseID}\ntspan:   \t{tspan}\nrendering:\t{renderingBool}")
    print(f"total_timesteps: {total_timesteps}")
    print(f"norm_reward: {norm_reward}; norm_obs = {norm_obs}")
    print(f"gamma:     \t{discountFactor}\nent_coef:\t{ent_coef}\nlearning_rate:\tlinear from {lr_schedule(1)} to {lr_schedule(0)}")
    print(f"vf_coef: \t{vf_coef}\n clip_range:\t{clip_range}\ngae_lambda:\t{gae_lambda}")
    print(f"n_steps:\t{n_steps*n_envs}\nbatch_size:\t{batch_size}\nn_epochs:\t{n_epochs}")

    print("***************************************************************************\n")

    print("please check the parameters and press enter to start the training...")
    input()

    # Reset the environment
    print("RESETTING THE ENVIRONMENT...")
    env.reset()

    # switch case to select between training new model and continue training the old one
    match trainingType:
        case "TRAIN_NEW_MODEL":
            # create the model
            model = PPO('MlpPolicy', env=env,
                        learning_rate=lr_schedule,
                        ent_coef=ent_coef,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        vf_coef=vf_coef,
                        clip_range=clip_range,
                        gae_lambda=gae_lambda,
                        device=deviceType, verbose=1, gamma=discountFactor, tensorboard_log=RLagent.log_dir) # USING GPU
        case "CONTINUE_TRAINING_OLD_MODEL":
            # definition of the learning parameters
            RLagentOLD = config.RL_config.recall(modelNameOLD,"latest") # recall latest trained model saved under the given model Name
            model = PPO.load(f"{RLagentOLD.model_dir}/{RLagentOLD.modelNumber}", env=env,
                            learning_rate=lr_schedule,
                            ent_coef=ent_coef,
                            n_steps=n_steps,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            vf_coef=vf_coef,
                            clip_range=clip_range,
                            gae_lambda=gae_lambda,
                            device=deviceType, verbose=1, tensorboard_log=RLagent.log_dir)
        case _:
            raise Exception("training Type not defined.")

    # save the starting model to ensure it will save
    model.save(RLagent.modelFileNameDir)
    
    # save a file containing the learning parameters used
    with open(f"{RLagent.model_dir}/training_parameters.txt", "w") as f:
        f.write("***************************************************************************\n")
        f.write("-- AGENT TRAINING PARAMETERS --\n")
        if trainingType == "TRAIN_NEW_MODEL":
            f.write(f"Training: {modelName} (new) on {deviceType}\n")
        else:
            f.write(f"Training: {modelName} (continue) from {modelNameOLD} on {deviceType}\n")
        f.write(f"Using {max_num_threads} threads, running {n_envs} environments in parallel.\n")
        f.write(f"Phase ID:\t{phaseID}\ntspan:   \t{tspan}\nrendering:\t{renderingBool}\n")
        f.write(f"total_timesteps: {total_timesteps}\n")
        f.write(f"norm_reward: {norm_reward}; norm_obs = {norm_obs}\n")
        f.write(f"gamma:     \t{discountFactor}\nent_coef:\t{ent_coef}\n")
        f.write(f"learning_rate:\tlinear from {lr_schedule(1)} to {lr_schedule(0)}\n")
        f.write(f"vf_coef: \t{vf_coef}\nclip_range:\t{clip_range}\ngae_lambda:\t{gae_lambda}\n")
        f.write(f"n_steps:\t{n_steps*n_envs}\nbatch_size:\t{batch_size}\nn_epochs:\t{n_epochs}\n")
        f.write(f"Total time steps: {total_timesteps}\n")
        f.write("***************************************************************************\n")
        f.write(f"Started Training: {datetime.now().strftime('%Y/%m/%d at %H:%M')}")
        f.close()
    ########################################################################################################
    ## TRAINING ##
    print("TRAINING ...")
    torch.set_num_threads(max_num_threads) # set the maximum threads available
    torch.set_num_interop_threads(max_num_threads)  #
    print(f"Using {max_num_threads} threads. Using {n_envs} in parallel.")

    # train the model
    try:
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True, tb_log_name=RLagent.modelName)
        print(f"FINISHED TRAINING: {datetime.now().strftime('%Y/%m/%d AT %H:%M')}")
    except Exception as e: # catch any exception during training
        print("EXCEPTION DURING TRAINING:")
        print(e)
        input("Press Enter to continue...")
        with open(f"log-{modelName}.txt", "a") as logfile:
            logfile.write(f"Encountered Error at: {datetime.now().strftime('%Y/%m/%d at %H:%M')}\n")
            logfile.write(f"  [EXCEPTION DURING TRAINING: {e}]\n")
            logfile.close()
            input("Press Enter to continue...")

    # save the model and the normalization statistics
    try:
        if norm_obs or norm_reward:
            # save the normalization if used
            with open(f"{RLagent.model_dir}/vec_normalize.pkl", "w") as file:
                env.save(f"{RLagent.model_dir}/vec_normalize.pkl")        # save the normalization
            printNormalization(RLagent)

        # save the model
        model.save(RLagent.modelFileNameDir)

    except Exception as e: # catch any exception during saving
        print("EXCEPTION DURING SAVING NORMALIZATION:")
        print(e)
        input("Press Enter to continue...")
        with open(f"log-{modelName}.txt", "a") as logfile:
            logfile.write(f"Encountered Error at: {datetime.now().strftime('%Y/%m/%d at %H:%M')}\n")
            logfile.write(f"  [EXCEPTION DURING SAVING NORMALIZATION: {e}]\n")
            logfile.close()
            input("Press Enter to continue...")


    #########################################################################################################

    ##### EXTRA #####

    # output to recall which model was trained in that window
    print("\n***************************************************************************")
    print("-- AGENT TRAINING PARAMETERS --")
    if trainingType == "TRAIN_NEW_MODEL":
        print(f"Training: {modelName} (new) on {deviceType}")
    else:
        print(f"Training: {modelName} (continue) from {modelNameOLD} on {deviceType}.")
    print(f"Phase ID:\t{phaseID}\ntspan:   \t{tspan}\nrendering:\t{renderingBool}")
    print(f"total_timesteps: {total_timesteps}")
    print(f"norm_reward: {norm_reward}; norm_obs = {norm_obs}")
    print(f"gamma:     \t{discountFactor}\nent_coef:\t{ent_coef}\nlearning_rate:\tlinear from {lr_schedule(1)} to {lr_schedule(0)}")
    print(f"vf_coef: \t{vf_coef}\n clip_range:\t{clip_range}")
    print(f"n_steps:\t{n_steps}\nbatch_size:\t{batch_size}\nn_epochs:\t{n_epochs}")
    print("***************************************************************************\n")

    with open(f"{RLagent.model_dir}/training_parameters.txt", "a") as f:
        f.write(f"Finished Training: {datetime.now().strftime('%Y/%m/%d at %H:%M')}\n")
        f.write(f"  [Trained on {max_num_threads} threads]\n")
        f.close()