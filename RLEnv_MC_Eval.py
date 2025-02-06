from SimEnvRL import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
#import matlab.engine
from datetime import datetime
#import sys
import argparse
# TODO: the initial target state is always exact periselene. It should be randomized.

# allow for terminal variables to be passed as arguments
# syntax:   python3 RLEnv_MC_Eval.py [phaseID] [n_samples] [agentName]
        
# Create Argument Parser
parser = argparse.ArgumentParser(description="Monte Carlo Analysis parameters")

# Add argument for reward normalization
parser.add_argument("-p", "--phase", type=int, default=2, help="Mission Phase")
parser.add_argument("-m", "--model", type=str, default="_NO_AGENT_", help="Model Name")
parser.add_argument("-s","--seed", type=str, default="None", help="Seed value used to initialize the simulations")
parser.add_argument("-n","--n-samples", type=int, default=1, help="number of Monte Carlo samples")
parser.add_argument("-r", "--render", type=str, default=True, help="Rendering bool")
# Parse arguments
argspar = parser.parse_args()

phaseID = argspar.phase
n_samples = argspar.n_samples
agentName = argspar.model
seed = argspar.seed
if argspar.render == "True":
    renderingBool  = True # rendering of the simulation
else:
    renderingBool  = False # rendering of the simulation

if seed != 'None':
    seed = int(seed)
else:
    seed = None

if agentName == "_NO_AGENT_":
    usingAgentBool = False
    print("Agent is NOT used to control the chaser. SIMULATING SAFE MODE.")
else:
    usingAgentBool = True
    print(f"Agent {agentName} is used to control the chaser.")

print(f"Running MC simulation (phase {phaseID})")
print(f"number of samples: {n_samples}")
print(f"Using seed: {seed}")
print(f"Rendering: {renderingBool}")
print("Please press enter to continue...")
input()
# agentName in the form: "Agent_P2-PPO-v12-achiral"
#if len(sys.argv) < 3:
#    phaseID = 2
#    n_samples = 1
#    usingAgentBool = True
#    agentName = "Agent_P2-PPO-v4.0-achiral-stable"
#    raise ValueError("Parameters not provided. Please use the syntax: python3 RLEnv_MC_Eval.py [phaseID] [n_samples] [agentName] [seed]")
#else:
#    phaseID = int(sys.argv[1])
#    n_samples = int(sys.argv[2])
#    if len(sys.argv) > 3: # means the agent is being used
#        agentName = sys.argv[3]
#        if agentName == "_NO_AGENT_":
#            usingAgentBool = False
#            print("Agent is NOT used to control the chaser. SIMULATING SAFE MODE.")
#            print("Please press enter to continue...")
#            input()
#        else:
#            usingAgentBool = True
#        seed = int(sys.argv[4]) if len(sys.argv) > 4 else None  # Optional seed
#        print(f"Agent {agentName} is used to control the chaser.")
#    else:
#        usingAgentBool = False 
#        print("Agent is NOT used to control the chaser. SIMULATING SAFE MODE.")
#        print("Please press enter to continue...")
#        input()
#        agentName = "_NO_AGENT_"

## GENREAL DATA
# phaseID = 2
# usingAgentBool = False 
# agentName = "_NO_AGENT_"

## ENVIROMENT PARAMETERS
if phaseID == 1:
    tspan = np.array([0, 0.04]) # ca 4 hours
elif phaseID == 2:
    tspan = np.array([0, 0.033]) # ca 3.3 hours


print("***************************************************************************")
print(f"Monte Carlo Analysis of {n_samples} samples. Agent: {agentName}")
print(f"Phase ID: {phaseID}, tspan: {tspan}, rendering: {renderingBool}")
if seed is not None:
    print(f"Using seed: {seed}")
else:
    print("No seed provided. Running with random initialization.")
print("***************************************************************************")

# MONTE CARLO PARAMETERS
n_samples_speed = None # if None generates all different speeds for each sample

## SCRIPT STARTS HERE ##
print("RUNNING A NEW MONTE CARLO SIMULATION ...")

# initialization of the environment
env = gym.make("SimEnv-v4.8", options={"phaseID":phaseID,"tspan":tspan,"renderingBool":renderingBool})
#env = DummyVecEnv([lambda: gym.make('SimEnv-v4.8',options={"phaseID": phaseID, "tspan": tspan, "renderingBool": renderingBool})])

if seed is not None:
    env.action_space.seed(seed)  # Seed the Gym action space
    env.observation_space.seed(seed)  # Seed the Gym observation space
# load the model
if usingAgentBool:
    RLagent = config.RL_config.recall(agentName,"latest")
    model = PPO.load(f"{RLagent.model_dir}/{RLagent.modelNumber}", env=env, device="cpu", seed=seed)

    try: # load the normalization stuff if normalization is true
        env = VecNormalize.load(f"{RLagent.model_dir}/vec_normalize.pkl", env)
        # Disable training mode to prevent statistics from updating
        env.training = False
        env.norm_reward = False
    except Exception as e:
        print(e)

print("GENERATING A POPULATION FOR THE SIMULATIONS... ",end='')
data : dict = {
        "seed": seed if seed is not None else "None",
        "phaseID" : env.unwrapped.param.phaseID,
        "agentModelName" : agentName,
        "n_population" : None,
        "param" : {
            "xc": env.unwrapped.param.xc,
            "tc": env.unwrapped.param.tc,
            "massRatio": env.unwrapped.param.massRatio,
            "freqGNC": env.unwrapped.param.freqGNC,
            "RLGNCratio": env.unwrapped.param.RLGNCratio,
            "chaserThrust": env.unwrapped.param.maxAdimThrust,
            "chaserMass": env.unwrapped.param.chaser["mass"],
            "chaserSpecificImpulse": env.unwrapped.param.specificImpulse,
            },
        "timeHistory" : np.arange(env.unwrapped.param.tspan[0], env.unwrapped.param.tspan[1] + (1/env.unwrapped.param.freqGNC), 1/env.unwrapped.param.freqGNC),
        "targetTrajectory_S" : None,
        "trueRelativeStateHistory_L" : None,
        "AgentAction" : None,
        "controlAction" : None,
        "constraintViolation" : None,
        "CPUExecTimeHistory" : None,
        "terminalState" : None,
        "terminalTimeIndex" : None,
        "fail" : None,
        "success" : None,
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
data["fail"] = np.zeros(data["n_population"])
data["success"] = np.zeros(data["n_population"])
data["targetTrajectory_S"] = np.zeros((len(data["timeHistory"])-1, 6, data["n_population"]))
data["trueRelativeStateHistory_L"] = np.zeros((len(data["timeHistory"])-1, 6, data["n_population"]))
data["controlAction"] = np.zeros((len(data["timeHistory"]), 3, data["n_population"]))
data["terminalState"] = np.zeros((6, data["n_population"]))
data["terminalTimeIndex"] = np.zeros(data["n_population"])
data["AgentAction"] = np.zeros((len(data["timeHistory"])-1, data["n_population"]))
data["OBoTUsage"] = np.zeros((len(data["timeHistory"])-1, data["n_population"]))
data["constraintViolation"] = np.zeros((len(data["timeHistory"])-1, data["n_population"]))
data["CPUExecTimeHistory"] = np.zeros((len(data["timeHistory"])-1, data["n_population"]))

# GENERATE THE POPULATION (states) - DEPENDING ON THE PHASE ID
match phaseID:
    case 2: # FOR PHASE ID 2
        if n_samples_speed is not None:
            val = {}
            val['R_BAR'] = -1 +2 * np.random.rand(1, n_samples)  # from -1 to +1 km
            val['V_BAR'] = -4.5 + 3 * np.random.rand(1, n_samples)   # from -4.5 to -1.5 km
            val['H_BAR'] = -1 + 2 * np.random.rand(1, n_samples) # from -1 to +1 km
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
    case 1: # FOR PHASE ID 1
        if n_samples_speed is None: # this generation runs multiple random conditions
            val = {}

            # Generate points in the annulus
            R_inner = 2  # minimum radius in km
            R_outer = 9  # maximum radius in km
            r = np.cbrt(np.random.uniform(R_inner**3, R_outer**3, n_ICs))  # Uniform in volume
            theta = np.random.uniform(0, np.pi, n_ICs)  # Polar angle
            phi = np.random.uniform(0, 2 * np.pi, n_ICs)  # Azimuthal angle

            # Convert to Cartesian coordinates
            val['R_BAR'] = r * np.sin(theta) * np.cos(phi)  # R component
            val['V_BAR'] = r * np.sin(theta) * np.sin(phi)  # V component
            val['H_BAR'] = r * np.cos(theta)  # H component

            # Generate random speeds
            val['speed_R_BAR'] = 1e-3 * (-5 + 10 * np.random.rand(1, n_ICs))  # Speed R component in km/s
            val['speed_V_BAR'] = 1e-3 * (-5 + 10 * np.random.rand(1, n_ICs))  # Speed V component in km/s
            val['speed_H_BAR'] = 1e-3 * (-5 + 10 * np.random.rand(1, n_ICs))  # Speed H component in km/s

            # Stack the population matrix
            POP = np.array([
                val['R_BAR'],
                val['V_BAR'],
                val['H_BAR'],
                val['speed_R_BAR'][0],
                val['speed_V_BAR'][0],
                val['speed_H_BAR'][0]
            ])
        else:
            raise ValueError("PHASE ID 1 NOT IMPLEMENTED YET")
    case _:
        raise ValueError("given phaseID not defined correctly")
    
# Adimensionalize the initial conditions
POP = POP / env.unwrapped.param.xc
POP[3:6, :] = POP[3:6, :] * env.unwrapped.param.tc

print("DONE.")

## RUN THE MONTE CARLO SIMULATION
print(f"STARTING MONTE CARLO SIMULATION... ESTIMATED TIME: {phaseID*data["n_population"]*4/60} [hours]")
start_time = time.time()

# RUN THE SIMULATIONS
fileNameSave = f"MC_P{phaseID}__{agentName}_{datetime.now().strftime('%Y_%m_%d_at_%H_%M')}.mat"

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
            if usingAgentBool:
                action, _ = model.predict(obs, deterministic=True) # predict the action using the agent
            else:
                action = 0 # if the agent is not used, the action is 0 (SKIP)
            obs, reward, terminated, truncated, info = env.step(action) # step
            if renderingBool:
                print(f"[sim:{sim_id+1}/{data["n_population"]}]",end='')

        tstartcomptime = time.time() - tstartcomptime
        print("COPYING THE SIMULATION INSIDE 'data' STRUCTURE: ",end='')
        # save the simulation data for future use:
        data["targetTrajectory_S"][:, :, sim_id + trgt_id] = env.unwrapped.fullStateHistory[:,0:6]
        data["trueRelativeStateHistory_L"][:, :, sim_id + trgt_id] = env.unwrapped.trueRelativeStateHistory_L
        data["controlAction"][:, :, sim_id + trgt_id] = env.unwrapped.controlActionHistory_L
        data["AgentAction"][:, sim_id + trgt_id] = env.unwrapped.AgentActionHistory
        data["OBoTUsage"][:, sim_id + trgt_id] = env.unwrapped.OBoTUsageHistory
        data["constraintViolation"][:, sim_id + trgt_id] = env.unwrapped.constraintViolationHistory
        data["terminalState"][:, sim_id + trgt_id] = env.unwrapped.terminalState
        data["terminalTimeIndex"][sim_id] = env.unwrapped.timeIndex
        data["fail"][sim_id + trgt_id] = 1 if env.unwrapped.terminationCause == "__CRASHED__" else 0
        data["success"][sim_id + trgt_id] = 1 if  env.unwrapped.terminationCause == "_AIM_REACHED_" else 0
        data["CPUExecTimeHistory"][:, sim_id + trgt_id] = env.unwrapped.CPUExecTimeHistory

        print(f" DONE.\n > Simulation Elapsed Time: {tstartcomptime/60:.2f} [min] ")

        # saving at each time step not to lose any of the simulation (in case of crash)
        print("SAVING THE SIMULATION: ",end='')
        # Save the Monte Carlo data to a .mat file
        if not os.path.exists("./Simulations/"):
            os.makedirs("./Simulations/")
        scipy.io.savemat(f"./Simulations/{fileNameSave}", {"data": data})
        print("DONE.\n")
        
print(f"\n >>> ALL SIMULATION DATA IS SAVED IN './Simulations/{fileNameSave}' <<<\n")