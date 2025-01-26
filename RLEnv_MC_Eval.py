from SimEnvRL import *
from stable_baselines3 import PPO
#import matlab.engine
from datetime import datetime

# TODO: the initial target state is always exact periselene. It should be randomized.

## ENVIROMENT PARAMETERS
phaseID = 2
tspan = np.array([0, 0.025])

# AGENT PARAMETERS
agentName = "Agent_P2-PPO-v10-achiral" # name of the agent to load
renderingBool  = True # rendering of the simulation

# MONTE CARLO PARAMETERS
n_samples = 1   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
n_samples_speed = None # if None generates all different speeds for each sample


## SCRIPT STARTS HERE ##
print("RUNNING A NEW MONTE CARLO SIMULATION ...")

# initialization of the environment
env = gym.make("SimEnv-v3", options={"phaseID":phaseID,"tspan":tspan,"renderingBool":renderingBool})

# load the model
RLagent = config.RL_config.recall(agentName,"latest")
model = PPO.load(f"{RLagent.model_dir}/{RLagent.modelNumber}", env=env, device="cpu")

print("Generating a population for the simulations...")
data : dict = {
        "phaseID" : env.unwrapped.param.phaseID,
        "param" : {
                    "xc": env.unwrapped.param.xc,
                    "tc": env.unwrapped.param.tc,
                    "massRatio": env.unwrapped.param.massRatio
                    },
        "timeHistory" : None,
        "trajectory" : None,
        "AgentAction" : None,
        "controlAction" : None,
        "constraintViolation" : None,
        "terminalState" : None,
        "fail" : None,
        "success" : None,
        "n_population" : None,
        "agentModelName" : agentName,
}

# uniform distribution for chaser position
if n_samples_speed:
    n_ICs = n_samples * n_samples_speed
else:
    n_ICs = n_samples


# target positions - see picture to understand the positions
initialStateTarget_S_batch = np.vstack([np.array([1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07])] # aposelene
                                                                                                # before aposelene
                                                                                                # after aposelene
                                                                                                # before periselene
                                                                                                # after periselene   
                                    )

n_targets_pos = initialStateTarget_S_batch.shape[0]
data["n_population"] = n_ICs + n_targets_pos - 1

# Initialize the variables for "faster" exec time
data["timeHistory"] = np.arange(env.unwrapped.param.tspan[0], env.unwrapped.param.tspan[1] + (1/env.unwrapped.param.freqGNC), 1/env.unwrapped.param.freqGNC)
data["fail"] = np.zeros(data["n_population"])
data["success"] = np.zeros(data["n_population"])
data["trajectory"] = np.zeros((len(data["timeHistory"])-1, 12, data["n_population"]))
data["controlAction"] = np.zeros((len(data["timeHistory"]), 3, data["n_population"]))
data["terminalState"] = np.zeros((6, data["n_population"]))
data["AgentAction"] = np.zeros((len(data["timeHistory"])-1, data["n_population"]))
data["OBoTUsage"] = np.zeros((len(data["timeHistory"])-1, data["n_population"]))
data["constraintViolation"] = np.zeros((len(data["timeHistory"])-1, data["n_population"]))

# Generate the population (states) - implemented 2 types of generation, the first one compares all the possible combinations
match phaseID:
    case 2: # FOR PHASE ID 2
        if n_samples_speed is not None:
            val = {}
            val['R_BAR'] = -0.5 + 1 * np.random.rand(1, n_samples)  # from -0.5 to +0.5 km
            val['V_BAR'] = -4 + 3 * np.random.rand(1, n_samples)   # from -4 to -1 km
            val['H_BAR'] = -0.5 + 1 * np.random.rand(1, n_samples) # from -0.5 to +0.5 km
            val['speed_R_BAR'] = 1e-3 * (-2 + 4 * np.random.rand(1, n_samples_speed)) # rand out in m/s, result in km/s
            val['speed_V_BAR'] = 1e-3 * (-2 + 4 * np.random.rand(1, n_samples_speed)) # rand out in m/s, result in km/s
            val['speed_H_BAR'] = 1e-3 * (-2 + 4 * np.random.rand(1, n_samples_speed)) # rand out in m/s, result in km/s
            
            POP = []
            for index_R in range(len(val['R_BAR'][0])):
                for index_V in range(len(val['V_BAR'][0])):
                    for index_H in range(len(val['H_BAR'][0])):
                        for index_speed_R in range(len(val['speed_R_BAR'][0])):
                            for index_speed_V in range(len(val['speed_V_BAR'][0])):
                                for index_speed_H in range(len(val['speed_H_BAR'][0])):
                                    POP.append([
                                        val['R_BAR'][0][index_R],
                                        val['V_BAR'][0][index_V],
                                        val['H_BAR'][0][index_H],
                                        val['speed_R_BAR'][0][index_speed_R],
                                        val['speed_V_BAR'][0][index_speed_V],
                                        val['speed_H_BAR'][0][index_speed_H]
                                    ])
            POP = np.array(POP).T  # Convert to NumPy array and transpose to match MATLAB's column format
        elif n_samples_speed is None: # this generation runs multiple random conditions
            val = {}
            val['R_BAR'] = -1 + 2 * np.random.rand(1, n_ICs)  # from -1 to +1 km
            val['V_BAR'] = -4 + 2 * np.random.rand(1, n_ICs)  # from -4 to -2 km
            val['H_BAR'] = -1 + 2 * np.random.rand(1, n_ICs)  # from -1 to +1 km
            val['speed_R_BAR'] = 1e-3 * (-2 + 4 * np.random.rand(1, n_ICs)) # rand out in m/s, result in km/s
            val['speed_V_BAR'] = 1e-3 * (-2 + 4 * np.random.rand(1, n_ICs)) # rand out in m/s, result in km/s
            val['speed_H_BAR'] = 1e-3 * (-2 + 4 * np.random.rand(1, n_ICs)) # rand out in m/s, result in km/s
            
            POP = np.array([
                val['R_BAR'][0],
                val['V_BAR'][0],
                val['H_BAR'][0],
                val['speed_R_BAR'][0],
                val['speed_V_BAR'][0],
                val['speed_H_BAR'][0]
            ])
    case 1:
        raise ValueError("PHASE ID 1 NOT IMPLEMENTED YET")
    case _:
        raise ValueError("given phaseID not defined correctly")
    
## RUN THE MONTE CARLO SIMULATION
print("Starting Monte Carlo analysis... (this operation can take several minutes)")
start_time = time.time()

# RUN THE SIMULATIONS

# Adimensionalize the initial conditions
POP = POP / env.unwrapped.param.xc
POP[3:6, :] = POP[3:6, :] * env.unwrapped.param.tc

fileNameSave = f"MC_run_{datetime.now().strftime('%Y_%m_%d_%H-%M-%S')}.mat"

# run the simulation for all the generated population
for trgt_id in range(n_targets_pos): # for each target position 
    initialStateTarget_S = initialStateTarget_S_batch[trgt_id,:]

    for sim_id in range(n_ICs): # for each initial condition
        tstartcomptime = time.time()
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print(f"\n\n############### RUNNING SIMULATION {sim_id + trgt_id +1} OUT OF {data["n_population"]} ###############")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # resetting the initial conditions and the environment
        terminated = False
        truncated = False
        obs, info = env.reset(options={"targetState_S": initialStateTarget_S,
                                    "relativeState_L": POP[:, sim_id + trgt_id].T})

        # run the simulation until termination/truncation:
        while ((not terminated) and (not truncated)):
            action, _ = model.predict(obs, deterministic=True) # predict the action using the agent
            obs, reward, terminated, truncated, info = env.step(action) # step
            if renderingBool:
                print(f"[sim:{sim_id+1}/{data["n_population"]}]",end='')

        tstartcomptime = time.time() - tstartcomptime
        # save the simulation data for future use:
        data["trajectory"][:, :, sim_id + trgt_id] = env.unwrapped.fullStateHistory
        data["controlAction"][:, :, sim_id + trgt_id] = env.unwrapped.controlActionHistory_L
        data["AgentAction"][:, sim_id + trgt_id] = env.unwrapped.AgentActionHistory
        data["OBoTUsage"][:, sim_id + trgt_id] = env.unwrapped.OBoTUsageHistory
        data["constraintViolation"][:, sim_id + trgt_id] = env.unwrapped.constraintViolationHistory
        data["terminalState"][:, sim_id + trgt_id] = env.unwrapped.terminalState
        data["fail"][sim_id + trgt_id] = 1 if env.unwrapped.terminationCause == "__CRASHED__" else 0
        data["success"][sim_id + trgt_id] = 1 if  env.unwrapped.terminationCause == "_AIM_REACHED_" else 0

        print(f" done > Elapsed time: {tstartcomptime/60:.2f} [min] ")

        # saving at each time step not to lose any of the simulation (in case of crash)
        print("SAVING THE SIMULATION: ",end='')
        # Save the Monte Carlo data to a .mat file
        if not os.path.exists("./Simulations/"):
            os.makedirs("./Simulations/")
        scipy.io.savemat(f"./Simulations/{fileNameSave}", {"data": data})
        print("DONE.\n")
        
print(f"\n >>> ALL SIMULATION DATA IS SAVED IN './Simulations/{fileNameSave}' <<<\n")