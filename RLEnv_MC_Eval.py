from SimEnvRL import *
from stable_baselines3 import PPO
import matlab.engine
from datetime import datetime

## PARAMETERS THAT CAN BE CHANGED:
phaseID = 2
tspan = [0, 0.015]

#  MONTE CARLO PARAMETERS
n_samples = 10   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
n_samples_speed = None # if None generates all different speeds for each sample

match "NEW_EVAL": # decide to "LOAD" or "NEW_EVAL" to load or re-execute MC simulation
    case "NEW_EVAL":
        print("RUNNING A NEW MONTE CARLO SIMULATION ...")
        # initialization of the environment
        env = gym.make("SimEnv-v1", options={"phaseID":phaseID,"tspan":tspan})

        # load the model
        RLagent = config.RL_config.recall("PhaseID_2-PPO_4","latest")
        model = PPO.load(RLagent.model_dir, env=env)

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
                "terminalState" : None,
                "fail" : None,
                "success" : None,
        }

        # target positions - see picture to understand the positions
        initialStateTarget_S_batch = np.vstack([np.array([1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07])] # aposelene
                                                                                                        # before aposelene
                                                                                                        # after aposelene
                                                                                                        # before periselene
                                                                                                        # after periselene   
                                            )

        # uniform distribution for chaser position
        if n_samples_speed:
            n_ICs = n_samples * n_samples_speed
        else:
            n_ICs = n_samples

        n_targets_pos = initialStateTarget_S_batch.shape[0]
        data["n_population"] = n_ICs + n_targets_pos - 1

        # Generate the population (states) - implemented 2 types of generation, the first one compares all the possible combinations
        match phaseID:
            case 2: # FOR PHASE ID 2
                if n_samples_speed is not None:
                    val = {}
                    val['R_BAR'] = -0.5 + 1 * np.random.rand(1, n_samples)  # from -0.5 to +0.5 km
                    val['V_BAR'] = -4 + 2 * np.random.rand(1, n_samples)   # from -4 to -2 km
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
        POP = POP / env.param.xc
        POP[3:6, :] = POP[3:6, :] * env.param.tc

        # Initialize the variables for "faster" exec time
        data["timeHistory"] = np.arange(env.param.tspan[0], env.param.tspan[1] + (1/env.param.freqGNC), 1/env.param.freqGNC)
        data["fail"] = np.zeros(data["n_population"])
        data["success"] = np.zeros(data["n_population"])
        data["trajectory"] = np.zeros((len(data["timeHistory"])-1, 12, data["n_population"]))
        data["terminalState"] = np.zeros((1, 6, data["n_population"]))

        # run the simulation for all the generated population
        for trgt_id in range(n_targets_pos): # for each target position 
            initialStateTarget_S = initialStateTarget_S_batch[trgt_id,:]

            for sim_id in range(n_ICs): # for each initial condition
                print(f" ## RUNNING SIMULATION {sim_id + trgt_id +1} OUT OF {data["n_population"]} ##")

                # resetting the initial conditions and the environment
                terminated = False
                truncated = False
                obs, info = env.reset(options={"targetState_S": initialStateTarget_S,
                                            "relativeState_L": POP[:, sim_id + trgt_id].T})

                # run the simulation until termination/truncation:
                while ((not terminated) and (not truncated)):
                    action, _ = model.predict(obs) # predict the action using the agent
                    obs, reward, terminated, truncated, info = env.step(action) # step
                    print(f"[sim: {sim_id}]",end='')
                    print(env.render())

                # save the simulation data for future use:
                data["trajectory"][:, :, sim_id + trgt_id] = env.unwrapped.fullStateHistory
                data["fail"][sim_id + trgt_id] = 1 if env.unwrapped.terminationCause == "__CRASHED__" else 0
                data["success"][sim_id + trgt_id] = 1 if env.unwrapped.terminationCause == "_DOCKING_SUCCESSFUL_" else 0

        print("SAVING THE MONTE CARLO SIMULATION: ",end='')
        # Save the Monte Carlo data to a .mat file
        scipy.io.savemat(f"Simulations/MC_run_{datetime.now().strftime('%Y_%m_%d_%H-%M-%S')}.mat", {"data": data})
        print("DONE.")

    case "LOAD":
        fileName = "MC_run_2025_01_15_11-22-14.mat"
        print(f"LOADING THE DATA FROM '{fileName}'")
        data = scipy.io.loadmat(f"Simulations/{fileName}")["data"]

# once the simulations are finished plot the results using matlab:
print("################## Running MATLAB ##################")

## MATLAB ##
# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Run the MATLAB script
eng.addpath('./matlabScripts', nargout=0)
eng.MonteCarloPlots(data, nargout=0)
# Add the directory containing the MATLAB scripts to the MATLAB path

# Stop MATLAB engine
eng.quit()