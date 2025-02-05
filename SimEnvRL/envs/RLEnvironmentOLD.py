import time
from SimEnvRL.generalScripts import *
from SimEnvRL import config
import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces

#import colorama
#colorama.init()

## REINFORCEMENT LEARNING selfRONMENT ##
class SimEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    ## INITIALIZATION ##
    def __init__(self, options=None):
        super(SimEnv,self).__init__()

        # propreties are:
        self.terminated: bool = False
        self.truncated: bool = False
        self.terminationCause: str = "None"

        # simulation initialization dataclasses
        self.initialValue: config.env_config.InitialValues = None

        # physical selfronment
        self.timeIndex: int = 0                                 # current time step index
        self.timeNow: float = 0.                                # current time
        self.targetState_S: np.ndarray = None                   # target state in Synodic (for physical environment)
        self.chaserState_S: np.ndarray = None                   # chaser state in Synodic (for physical environment)

        # ON BOARD data
        self.OBoptimalTrajectory: dict = None                   # current OB optimal Trajectory (latest)
        self.OBStateTarget_M: np.ndarray = None                 # OB estimated Target State, in Moon Synodic [M]
        self.OBStateRelative_L: np.ndarray = None               # OB estimated relative state in LVLH reference frame
        self.observation: np.ndarray = None                     # Agent observation (see definition to know more)

        # Historical values of the simulation
        self.timeHistory: np.ndarray = None                     # time stamp for each step
        self.controlActionHistory_L: np.ndarray = None          # array containing the control action required (adimensional)
        self.fullStateHistory: np.ndarray = None                # array containing the full state history (target + chaser) in Synodic
        self.terminalState: np.ndarray = None                   # array containing the terminal state of chaser in LVLH (real state not OB!)
        self.AgentActionHistory: np.ndarray = None              # sequence of the AgentActions (0,1,2)
        self.constraintViolationHistory: np.ndarray = None      # boolean if constraint violations occurred during the simulation
        self.OBoTUsageHistory : np.ndarray = None               # whether the OBoptimalTrajectory was used (this can highlight possible failures in ASRE L1)

        # create the selfronment simulation parameters dataclass
        options = options or {}

        # if rendering is required
        if "renderingBool" in options:
            self.renderingBool = options.get("renderingBool", False)
        else:
            self.renderingBool = False  # default value is no rendering

        # check "phaseID"
        if "phaseID" not in options:
            raise AttributeError("The 'phaseID' key is required in the options to start the environment.")
        # check "tspan"
        if "tspan" not in options or options["tspan"] is None:
            raise AttributeError("'tspan' is required but was not provided. Please set it explicitly.")
        # check 'tspan' type is np.ndarray
        elif not isinstance(options["tspan"], np.ndarray) and not options["tspan"].size == 2:
            raise TypeError("'tspan' must be a numpy ndarray.")
        
        if options:
            options = {
                "phaseID": options.get("phaseID"),
                "tspan": options["tspan"] if isinstance(options["tspan"], np.ndarray) else None
            }
        else:
            raise AttributeError("options are required to start the environment. Please check that 'phaseID' and 'tspan' are correctly defined.")
        self.param = config.env_config.getParam(phaseID=options["phaseID"],tspan=options["tspan"])

        ## OBSERVATION SPACE
        # (the first 6 values are OBStateRelative_L, 
        # the last one is OBoTAge) NOTE THAT OBoTAge is expressed is ADIMENSIONAL in the observation space
        self.observation_space = spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1,
                                                          -1]),
                                            high=np.array([1, 1, 1, 1, 1, 1,
                                                           1]),
                                            dtype=np.float64)

        ## ACTION SPACE
        # 2 actions are present : 0 [skip Loop 1] or 1 [compute Loop 1] or 2 [delete OBoptimalTrajectory]
        self.action_space = spaces.Discrete(3)

    ## STEP ##
    def step(self, AgentAction):

        RLstepAgentAction = AgentAction
        GNCiterID = 0
        while GNCiterID < self.param.RLGNCratio: # loop for the GNC cycles
            GNCiterID += 1
        # execute "RLGNCratio" times the GNC and then let the agent decide following action
        # note that after the first GNC cycle the AgentAction is set to 0 (SKIP) for the following cycles until a new action is taken
            if (self.timeIndex < len(self.timeHistory)-1): # check if the simulation is not over (still space inside the History vectors)
                # extract parameters for the current time step
                self.timeNow = self.timeHistory[self.timeIndex]
                self.AgentActionHistory[self.timeIndex] = AgentAction
                
                # NAVIGATION # NOTE: this has already been computed for the current time step in previous cycle
                # indeed, the NAVIGATION is required for the agent to determine its action
                # self.OBStateTarget_M, _, self.OBStateRelative_L = OBNavigation(self.targetState_S, self.chaserState_S, self.param)

                # GUIDANCE ALGORITHM # 
                # compute the control action and output the optimal trajectory (if re-computed)

                # compute the Age of the OB optimal Trajectory, if the optimal Trajectory exists, before applying the AgentAction
                if self.OBoptimalTrajectory:
                    # OBoTAge is expressed in HOURS!!!
                    OBoTAge = (self.timeNow - self.OBoptimalTrajectory["envStartTime"]) * self.param.tc/3600 # in hours
                else:
                    OBoTAge = -1 # default value to say that the optimal trajectory does not exist

                #executionTime_start = time.time()
                controlAction_L, self.OBoptimalTrajectory = \
                                        OBGuidance(self.timeNow, self.OBStateRelative_L, self.OBStateTarget_M,
                                            self.param.phaseID, self.param, AgentAction, self.OBoptimalTrajectory)       
                #executionTime = time.time() - executionTime_start
                #print(f"  > Guidance Step Execution Time: {executionTime*1e3:.2f} [ms]")
                if self.OBoptimalTrajectory: # save if for the current time step the optimal trajectory was used
                    self.OBoTUsageHistory[self.timeIndex] = True
                else:
                    self.OBoTUsageHistory[self.timeIndex] = False

                # CONTROL ACTION #
                
                # rotate the control action from the local frame to the synodic frame
                controlAction_S, controlAction_L = OBControl(self.targetState_S,controlAction_L,self.param)
                self.controlActionHistory_L[self.timeIndex+1,:] = controlAction_L

                # PHYSICAL selfRONMENT #
                # propagate the dynamics of the chaser for one time step (depends on Guidance Frequency)
                distAcceleration_S = ReferenceFrames.computeEnvironmentDisturbances(self.timeNow, self.param.chaser, self.param)
                odesol = solve_ivp(lambda t, state: dynamicsModel.CR3BP(t, state, self.param, controlAction_S, distAcceleration_S),
                                    [self.timeNow, self.timeHistory[self.timeIndex + 1]], self.chaserState_S,
                                    method="DOP853", rtol=1e-9, atol=1e-8)
                self.fullStateHistory[self.timeIndex+1, 6:12] = odesol.y[:,-1] # extract following time step

                # PREPARE FOR NEXT TIME STEP #
                self.timeIndex += 1
                self.targetState_S = self.fullStateHistory[self.timeIndex,:6]
                self.chaserState_S = self.fullStateHistory[self.timeIndex,6:12]
                self.OBStateTarget_M, _, self.OBStateRelative_L = OBNavigation(self.targetState_S, self.chaserState_S, self.param)
                AgentAction = 0 # reset the AgentAction to SKIP (0) for the next GNC loop; note that the AgentAction shall only be applied once
                self.truncated = False
            else:
                GNCiterID = self.param.RLGNCratio + 1 # to end the GNC loop
                # END OF SIMULATION # (out of time)
                # self.truncated = self.EOS(self.timeHistory[self.timeIndex],self.param)        
                self.terminationCause = "_OUT_OF_TIME_"
                print("\n <<<<<<<<<<<<<<< OUT OF TIME >>>>>>>>>>>>>>> \n")
                self.truncated = True
            # ---- end of GNC loop ----- #

        # RL AGENT OBSERVATION #
        self.observation = self.computeRLobservation()

        # REWARD COMPUTATION #
        self.stepReward, self.terminated = self.computeReward(RLstepAgentAction, OBoTAge, controlAction_L,
                                                              self.param.phaseID, self.param)
        
        # rendering the environment if required
        if self.renderingBool:
            print(self.render())
            
        info = {} #{"param": self.param, "timeNow": self.timeNow}

        return self.observation, self.stepReward, self.terminated, self.truncated, info

    ## RENDER ##
    def render(self, mode='ansi'):
        if mode != 'ansi':
            raise ValueError("Unsupported render mode. Supported mode: 'ansi'")
        
        if self.timeIndex == 0:
            return "Environment just started. No actions yet."
        
        # Define ANSI escape codes for colors
        colors = {
            "yellow": "\033[93m",  # Yellow
            "green": "\033[92m",   # Green
            "red": "\033[91m",     # Red
            "reset": "\033[0m"     # Reset to default
        }
        
        # Retrieve the agent's action from history
        action = self.AgentActionHistory[self.timeIndex - self.param.RLGNCratio]
        action_text = {0: "SKIP", 1: "COMPUTE", 2: "DELETE"}.get(action, "UNKNOWN")
        color = colors["reset"]  # Default color
        
        # Assign color based on the action and conditions
        if action == 0:  # SKIP
            color = colors["green"] if self.OBoptimalTrajectory is not None else colors["red"]
        elif action == 1:  # COMPUTE
            color = colors["yellow"]
        elif action == 2:  # DELETE
            color = colors["red"]
        else:
            raise ValueError("Agent Action Not Defined.")
        
        # Format the output string with color
        ansi_output = f"[envTime = {self.timeNow:.5f}] AgentAction: {color}{action_text}{colors['reset']} (reward = {self.stepReward})"
        
        return ansi_output


    ## RESET ##
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # RL related parameters
        self.terminated = False
        self.truncated = False
        self.terminationCause = "Unknown"
        self.stepReward = 0.

        # physical selfronment related parameters
        self.timeIndex = 0
        self.timeNow = 0.

        # guidance related parameters
        self.OBoptimalTrajectory = None

        # defining the initial values
        self.initialValue, typeOfInitialConditions = config.env_config.getInitialValues(self.param,seed,options)

        # definition of the time vector with the GNC frequency
        self.timeHistory = np.arange(self.param.tspan[0], self.param.tspan[-1], 1/(self.param.freqGNC))

        # INITIALIZATION OF THE MAIN VALUES FOR FULL SIMULATION HISTORY (definition of the solution vectors)
        self.controlActionHistory_L = np.zeros((len(self.timeHistory)+1, 3))
        self.fullStateHistory = np.zeros((len(self.timeHistory),12))
        self.AgentActionHistory = np.zeros((len(self.timeHistory),))
        self.constraintViolationHistory = np.zeros((len(self.timeHistory),))
        self.OBoTUsageHistory = np.zeros((len(self.timeHistory),)).astype(bool)

        # extraction of the initial conditions
        self.targetState_S = self.initialValue.fullInitialState[0:6]
        self.chaserState_S = self.initialValue.fullInitialState[6:12]

        # saving the initial states inside the fullStateHistory vector
        self.fullStateHistory[0,:] = self.initialValue.fullInitialState

        # integrate the dynamics of the target [for the whole simulation time]
        distAcceleration_S = dynamicsModel.computeEnvironmentDisturbances(0,self.param.target,self.param)
        odesol = solve_ivp(lambda t, state: dynamicsModel.CR3BP(t, state, self.param, distAcceleration_S),
                                [self.timeHistory[0], self.timeHistory[-1]], self.targetState_S, t_eval=self.timeHistory,
                                method="DOP853", rtol=1e-8, atol=1e-7)
        self.fullStateHistory[:, :6] = odesol.y.T # store the target dynamics

        ## compute RL Agent Observation at time step 1
        self.OBStateTarget_M, _, self.OBStateRelative_L = OBNavigation(self.targetState_S, self.chaserState_S, self.param)

        info = {"initialConditionsUsed": typeOfInitialConditions}
        return self.computeRLobservation(), info

    ## EXTRA METHODS ##
    def computeRLobservation(self):
        # compute the Age of the OB optimal Trajectory, if the optimal Trajectory exists, set to -1 if not
        if self.OBoptimalTrajectory and "envStartTime" in self.OBoptimalTrajectory:
            trajAGE = self.timeNow - self.OBoptimalTrajectory["envStartTime"]
        else:
            trajAGE = -1 # setting to -1 if the optimal trajectory does not exist

        # the observation is composed by the relative state in LVLH, and the trajectory age
        observation = np.hstack([self.OBStateRelative_L, trajAGE])

        return observation

    def computeReward(self, AgentAction, OBoTAge, controlAction, phaseID, param):
        terminated = False

        # compute the TRUE relative state in synodic and LVLH
        TRUE_relativeState_S = self.chaserState_S - self.targetState_S
        TRUE_relativeState_L = ReferenceFrames.convert_S_to_LVLH(self.targetState_S,TRUE_relativeState_S,param)

        # translate to meters the relative state, since all the constraints are defined in meters
        TRUE_relativeState_L_meters = TRUE_relativeState_L*param.xc*1e3 # m
        TRUE_relativeState_L_meters[3:6] /= param.tc # m/s

        # check if the constraints are violated and the "entity" of the violation
        constraintViolationBool, violationEntity = check.constraintViolation(TRUE_relativeState_L_meters, 
                                                            param.constraint["constraintType"],
                                                            param.constraint["characteristicSize"], param)
        self.constraintViolationHistory[self.timeIndex] = constraintViolationBool # save in the history if constraints are violated
    
        # check if the aim is reached
        aimReachedBool, crashedBool = check.aimReached(TRUE_relativeState_L, param.constraint["aimAtState"], self.param)

        # REWARD COMPUTATION DEPENDING ON THE PHASE ID #
        match phaseID:
            case 1: # RENDEZVOUS
                K_trigger = 1e-4
                K_deleted = 1
                K_control = 0.2
                K_precisn = 0.5
                K_simtime = 0.01
    
                ## ## ## ## ## ## ## ## ## ## REWARD COMPUTATION ## ## ## ## ## ## ## ## ## ##
                self.stepReward = 0.

                # Triggering Reward - Penalize frequent, unnecessary recomputation of trajectories
                match AgentAction:
                    case 0: # no action means no reward nor penalization
                        pass
                    case 1: # in case of a trajectory recomputation, give a small negative, according to the age of the trajectory
                        # this is to disincentive a continuous computation of the optimal trajectory (lower penality if old trajectory)
                        self.stepReward -= K_trigger * np.exp(-120*OBoTAge)
                    case 2: # if the agent deletes the optimal trajectory
                        if OBoTAge>=0:
                            # if the trajectory exists, the reward is reduced according to the age of the trajectory (lower penality if old trajectory)
                            self.stepReward -= K_deleted * np.exp(-10*OBoTAge)
                        else: # avoid "deleting" an inexistant trajectory
                            self.stepReward -= 1
                    case _:
                        pass

                # Precision Reward - give a positive reward for collision avoidance
                constraintFactor = abs(violationEntity) # observe that if a constraint is violated this reward turns to negative!
                proximityFactor = 0.5 * (1 + np.tanh(param.constraint["characteristicSize"] - np.linalg.norm(TRUE_relativeState_L_meters[0:3])))
                self.stepReward -= K_precisn * (constraintFactor) * proximityFactor

                # Time of Flight - penalize long time of flights
                self.stepReward -= K_simtime * 1/param.freqGNC  

                ## Fuel Efficiency Reward - Penalize large control actions
                # reduce the reward of an amount proportional to the Guidance control effort
                self.stepReward -= K_control * (1 - np.exp(-np.linalg.norm(controlAction)/self.param.maxAdimThrust)**2)

                ## Maximum Control Action Reward - Penalize control actions that exceed the maximum available
                if controlAction[0] > param.maxAdimThrust \
                or controlAction[1] > param.maxAdimThrust \
                or controlAction[2] > param.maxAdimThrust:
                    self.stepReward -= .5 # penalize the agent for exceeding the maximum control action

            case 2: # APPROACH AND DOCKING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # reward tunable parameters 
                K_trigger = 1e-4
                K_deleted = 1
                K_control = 0.01
                K_precisn = 1
                K_simtime = 0.01

                ## ## ## ## ## ## ## ## ## ## REWARD COMPUTATION ## ## ## ## ## ## ## ## ## ##
                self.stepReward = 0.

                # Triggering Reward - Penalize frequent, unnecessary recomputation of trajectories
                match AgentAction:
                    case 0: # no action means no reward nor penalization
                        pass
                    case 1: # in case of a trajectory recomputation, give a small negative, according to the age of the trajectory
                        # this is to disincentive a continuous computation of the optimal trajectory (lower penality if old trajectory)
                        self.stepReward -= K_trigger * np.exp(-120*OBoTAge)
                    case 2: # if the agent deletes the optimal trajectory
                        if OBoTAge>=0:
                            # if the trajectory exists, the reward is reduced according to the age of the trajectory (lower penality if old trajectory)
                            self.stepReward -= K_deleted * np.exp(-10*OBoTAge)
                        else: # avoid "deleting" an inexistant trajectory
                            self.stepReward -= 1
                    case _:
                        pass

                # Precision Reward - give a positive reward for good convergence
                if TRUE_relativeState_L[1] <= 0:
                    proximityFactor = np.exp(TRUE_relativeState_L_meters[1]/30) # the closer to the target on V BAR
                else:
                    proximityFactor = 1 # ceiling value for the proximity factor to avoid "RuntimeWarning: overflow encountered in exp"
                precisionFactor = -violationEntity # observe that if a constraint is violated this reward turns to negative!
                velocityFactor  = np.exp(-np.linalg.norm(TRUE_relativeState_L[3:6]-param.constraint["aimAtState"][3:6])) 
                self.stepReward += K_precisn * (precisionFactor + velocityFactor) * proximityFactor

                # Time of Flight - penalize long time of flights
                timeExpenseFactor = 1/param.freqGNC 
                proximityFactor = 1 - np.exp( - np.linalg.norm(TRUE_relativeState_L_meters[0:3]) / 3e3)**2 # the closer to the target
                self.stepReward -= K_simtime * timeExpenseFactor * proximityFactor 

                ## Fuel Efficiency Reward - Penalize large control actions
                # reduce the reward of an amount proportional to the Guidance control effort
                self.stepReward -= K_control * (1 - np.exp(-np.linalg.norm(controlAction)/self.param.maxAdimThrust)**2)

                ## Maximum Control Action Reward - Penalize control actions that exceed the maximum available
                if controlAction[0] > param.maxAdimThrust \
                or controlAction[1] > param.maxAdimThrust \
                or controlAction[2] > param.maxAdimThrust:
                    self.stepReward -= .5 # penalize the agent for exceeding the maximum control action
#
            case _:
                raise ValueError("reward function for this phaseID has not been implemented yet")
        
        ## Docking Successful / Aim Reached - reached goal :)
        if aimReachedBool:
            if phaseID == 1:
                print(" ################################# ")
                print(" >>>>> SUCCESSFUL RENDEZVOUS <<<<< ")
                print(" ################################# ")
                self.terminationCause = "_AIM_REACHED_"
            elif phaseID == 2:
                print(" ################################## ")
                print(" >>>>>>> SUCCESSFUL DOCKING <<<<<<< ")
                print(" ################################## ")
                self.terminationCause = "_AIM_REACHED_"
            terminated = True
            self.stepReward += 1
            self.terminalState = TRUE_relativeState_L

        ## Crash Reward - crash into the target
        elif crashedBool:
            print(" ################################### ")
            print(" ############# CRASHED ############# ")
            print(" ################################### ")
            terminated = True
            self.stepReward -= 1
            self.terminationCause = "__CRASHED__"
            self.terminalState = TRUE_relativeState_L


        # TODO: here normalize the reward

        return self.stepReward, terminated
    
    ## END OF SIMULATION ##
    def EOS(self,timeNow,param):
        # determine if the simulation run out of time [reached final tspan]
        if timeNow+1/param.freqGNC >= param.tspan[-1]:
            truncated = True
            self.terminationCause = "_OUT_OF_TIME_"
            print(" <<<<<<<<<<<<<<< OUT OF TIME >>>>>>>>>>>>>>> ")
        else:
            truncated = False

        return truncated
    
    ## GET THE HISTORY ##
    def getHistory(self):
        savedDictionary = {
            "phaseID": self.param.phaseID,
            "timeHistory" : self.timeHistory[0:self.timeIndex],
            "fullStateHistory" : self.fullStateHistory[0:self.timeIndex],
            "controlActionHistory_L" : self.controlActionHistory_L[0:self.timeIndex],
            "AgentActionHistory" : self.AgentActionHistory[0:self.timeIndex],
            "constraintViolationHistory" : self.constraintViolationHistory[0:self.timeIndex],
            "OBoTUsageHistory" : self.OBoTUsageHistory[0:self.timeIndex],
            "terminationCause" : self.terminationCause,
            "param": {
                "xc": self.param.xc,
                "tc": self.param.tc,
                "target" : self.param.target,
                "chaser" : self.param.chaser,
                "maxAdimThrust" : self.param.maxAdimThrust,
                "holdingState" : self.param.holdingState,
                "dockingState" : self.param.dockingState,
                "freqGNC" : self.param.freqGNC,
                "SolarFlux" : self.param.SolarFlux,
                "massRatio" : self.param.massRatio
            }
        }
        return savedDictionary