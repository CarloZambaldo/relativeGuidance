import time
from SimEnvRL.generalScripts import *
from SimEnvRL import config
import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces

import colorama
colorama.init()

## REINFORCEMENT LEARNING selfRONMENT ##
class SimEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, options=None):
        super(SimEnv,self).__init__()

        # propreties are:
        self.terminated: bool = False
        self.truncated: bool = False
        self.terminationCause: str = "None"

        # simulation initialization dataclasses
        self.initialValue: config.env_config.InitialValues = None

        # physical selfronment
        self.timeIndex: int = 0
        self.timeNow: float = 0.
        self.targetState_S: np.ndarray = None
        self.chaserState_S: np.ndarray = None

        # ON BOARD data
        self.OBoptimalTrajectory: dict = {}
        self.OBStateTarget_M: np.ndarray = None
        self.OBStateRelative_L: np.ndarray = None
        self.observation: np.ndarray = None

        # Historical values of the simulation
        self.timeHistory: np.ndarray = None
        self.controlActionHistory_L: np.ndarray = None
        self.fullStateHistory: np.ndarray = None
        self.AgentActionHistory: np.ndarray = None
        self.constraintViolationHistory: np.ndarray = None

        # create the selfronment simulation parameters dataclass
        options = options or {}
        if options:
            options = {
                "phaseID": options.get("phaseID"),
                "tspan": options["tspan"] if isinstance(options["tspan"], np.ndarray) else None
            }
        else:
            raise AttributeError("options are required to start the environment. Please check that 'phaseID' and 'tspan' are correctly defined.")
        self.param = config.env_config.getParam(phaseID=options["phaseID"],tspan=options["tspan"])

        ## OBSERVATION SPACE
        # (the first 6 values are OBStateRelative_L, the last one is OBoTAge)
        self.observation_space = spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1,  -1]),
                                            high=np.array([1, 1, 1, 1, 1, 1,   5]),
                                            dtype=np.float64)

        ## ACTION SPACE
        # 2 actions are present : 0 [skip Loop 1] or 1 [compute Loop 1] or 2 [delete OBoptimalTrajectory]
        self.action_space = spaces.Discrete(3)

    ## STEP ##
    def step(self, AgentAction):

        # extract parameters for the current time step
        self.timeNow = self.timeHistory[self.timeIndex]
        self.AgentActionHistory[self.timeIndex] = AgentAction
        
        # NAVIGATION # NOTE: this has already been computed for the current time step in previous cycle
        # indeed, the NAVIGATION is required for the agent to determine its action
        # self.OBStateTarget_M, _, self.OBStateRelative_L = OBNavigation(self.targetState_S, self.chaserState_S, self.param)

        # GUIDANCE ALGORITHM # 
        # compute the control action and output the optimal trajectory (if re-computed)
        #executionTime_start = time.time()
        controlAction_L, self.OBoptimalTrajectory = \
                                OBGuidance(self.timeNow, self.OBStateRelative_L, self.OBStateTarget_M,
                                    self.param.phaseID, self.param, AgentAction, self.OBoptimalTrajectory)        
        #executionTime = time.time() - executionTime_start
        #print(f"  > Guidance Step Execution Time: {executionTime*1e3:.2f} [ms]")

        # CONTROL ACTION #
        self.controlActionHistory_L[self.timeIndex+1,:] = controlAction_L
        # rotate the control action from the local frame to the synodic frame
        controlAction_S = OBControl(self.targetState_S,controlAction_L,self.param)

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
        self.observation = self.computeRLobservation()

        # END OF SIMULATION # (out of time)
        self.truncated = self.EOS(self.timeHistory[self.timeIndex],self.param)
        
        # REWARD COMPUTATION #
        self.stepReward, self.terminated = self.computeReward(AgentAction,controlAction_L,self.param.phaseID,self.param)

        print(self.render())

        info = {"param": self.param, "timeNow": self.timeNow}

        return self.observation, self.stepReward, self.terminated, self.truncated, info


    def render(self, mode='ansi'):
        if mode != 'ansi':
            raise ValueError("Unsupported render mode. Supported mode: 'ansi'")
        
        if self.timeIndex == 0:
            return "Environment just started. No actions yet."
        
        # Define colors using ANSI escape codes
        colors = {
            "yellow": "\033[93m",  # Yellow
            "green": "\033[92m",   # Green
            "red": "\033[91m",     # Red
            "reset": "\033[0m"     # Reset to default
        }
        
        # Determine the action and color based on conditions
        action = self.AgentActionHistory[self.timeIndex - 1]
        action_text = {0: "SKIP", 1: "COMPUTE", 2: "DELETE"}.get(action, "UNKNOWN")
        color = colors["reset"]  # Default color

        if action == 0:  # SKIP
            if self.OBoptimalTrajectory is not None:
                color = colors["green"]  # Green if optimal trajectory exists
            else:
                color = colors["red"]  # Red if optimal trajectory is None
        elif action == 1:  # COMPUTE
            color = colors["yellow"]  # Yellow for COMPUTE
        elif action == 2:  # DELETE
            color = colors["red"]  # Red for DELETE
        else:
            raise ValueError("Agent Action Not Defined.")
        
        # Format the output string with color
        ansi_output = f"[envTime = {self.timeNow:.5f}] AgentAction: {color}{action_text}{colors['reset']} (reward = {self.stepReward})"

        return ansi_output



    # The reset method should set the initial state of the selfronment (e.g., relative position and velocity) and return the initial observation.
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
        self.OBoptimalTrajectory = {}

        # defining the initial values
        ## ORIGINALLY WAS: self.param, self.initialValue = config.env_config.get(seed)
        self.initialValue, typeOfInitialConditions = config.env_config.getInitialValues(self.param,seed,options)

        # definition of the time vector with the GNC frequency
        self.timeHistory = np.arange(self.param.tspan[0], self.param.tspan[-1], 1/(self.param.freqGNC))

        # INITIALIZATION OF THE MAIN VALUES FOR FULL SIMULATION HISTORY (definition of the solution vectors)
        self.controlActionHistory_L = np.zeros((len(self.timeHistory)+1, 3))
        self.fullStateHistory = np.zeros((len(self.timeHistory),12))
        self.AgentActionHistory = np.zeros((len(self.timeHistory),))
        self.constraintViolationHistory = np.zeros((len(self.timeHistory),))

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
        if self.OBoptimalTrajectory and "envStartTime" in self.OBoptimalTrajectory:
            trajAGE = self.OBoptimalTrajectory["envStartTime"] - self.timeNow
        else:
            trajAGE = -1 # setting to -1 if the optimal trajectory does not exist

        observation = np.hstack([self.OBStateRelative_L, trajAGE])

        return observation



    def computeReward(self, AgentAction, controlAction, phaseID, param):

        terminated = False

        match phaseID:
            case 1:
                raise ValueError("reward function for this phaseID has not been implemented yet")
    
            case 2: # APPROACH AND DOCKING
                # reward tunable parameters 
                K_trigger = 0.0001
                K_deleted = 0.001
                K_cnstrnt = 0
                K_control = 0 #.001
                K_precisn = 10
                K_simtime = 1 #0.005

                # COMPUTE: check constraints and terminal values
                TRUE_relativeState_L = ReferenceFrames.convert_S_to_LVLH(self.targetState_S,self.chaserState_S-self.targetState_S,param)
                constraintViolationBool, violationEntity = check.constraintViolation(TRUE_relativeState_L, 
                                                                    param.constraint["constraintType"],
                                                                    param.constraint["characteristicSize"], param)
                
                aimReachedBool, crashedBool = check.aimReached(TRUE_relativeState_L, param.constraint["aimAtState"], self.param)
                
                self.constraintViolationHistory[self.timeIndex] = constraintViolationBool

                # compute the Age of the OB optimal Trajectory
                if self.OBoptimalTrajectory:
                    OBoTAge = (self.OBoptimalTrajectory["envStartTime"]-self.timeNow)
                else:
                    OBoTAge = -1


                ## ## ## ## ## ## ## ## ## ## REWARD COMPUTATION ## ## ## ## ## ## ## ## ## ##
                self.stepReward = 0.

                # Triggering Reward - Penalize frequent, unnecessary recomputation of trajectories
                match AgentAction:
                    case 0: # no action means no reward nor penalization
                        pass
                    case 1: # in case of a trajectory recomputation, give a small negative, according to the age of the trajectory
                        # this is to disincentive a continuous computation of the optimal trajectory (lower penality if old trajectory)
                        self.stepReward -= K_trigger/(1+1e4*OBoTAge)
                    case 2: # if the agent deletes the optimal trajectory
                        if self.OBoptimalTrajectory:
                            # if the trajectory exists, the reward is reduced according to the age of the trajectory (lower penality if old trajectory)
                            self.stepReward -= K_deleted/(1+1e3*OBoTAge)
                        else: # avoid "deleting" an inexistant trajectory
                            self.stepReward -= 100
                    case _:
                        pass

                # Precision Reward - give a positive reward for good convergence
                if TRUE_relativeState_L[1] <= 0:
                    proximityFactor = np.exp(TRUE_relativeState_L[1]*3*self.param.xc) # the closer to the target on V BAR
                else:
                    proximityFactor = 1 # ceiling value for the proximity factor to avoid "RuntimeWarning: overflow encountered in exp"
                    
                precisionFactor = - violationEntity # observe that if a constraint is violated this reward turns to negative!
                self.stepReward += K_precisn * precisionFactor * proximityFactor

                # Collision Avoidance Reward - Penalize proximity to obstacles (constraints violation)
                if constraintViolationBool:
                    self.stepReward -= K_cnstrnt * 10

                # Time of Flight - penalize long time of flights
                self.stepReward -= 1/param.freqGNC * K_simtime

                # Fuel Efficiency Reward - Penalize large control actions
                # reduce the reward of an amount proportional to the Guidance control effort
                self.stepReward -= K_control * np.linalg.norm(controlAction)

                # Crash Reward - crash into the target
                if crashedBool:
                    print(" ################################### ")
                    print(" ############# CRASHED ############# ")
                    print(" ################################### ")
                    terminated = True
                    self.stepReward -= 100
                    self.terminationCause = "__CRASHED__"
                
                # Docking Successful - reached goal :)
                if aimReachedBool:
                    print(" ################################## ")
                    print(" >>>>>>> SUCCESSFUL DOCKING <<<<<<< ")
                    print(" ################################## ")
                    terminated = True
                    self.stepReward += 500
                    self.terminationCause = "_DOCKING_SUCCESSFUL_"

            case _:
                raise ValueError("reward function for this phaseID has not been implemented yet")
        
        return self.stepReward, terminated
    

    def EOS(self,timeNow,param):
        # determine if the simulation run out of time [reached final tspan]
        if timeNow+1/param.freqGNC >= param.tspan[-1]:
            truncated = True
            self.terminationCause = "_OUT_OF_TIME_"
            print(" <<<<<<<<<<<<<<< OUT OF TIME >>>>>>>>>>>>>>> ")
        else:
            truncated = False

        return truncated
    
    def getHistory(self):
        savedDictionary = {
            "phaseID": self.param.phaseID,
            "timeHistory" : self.timeHistory[0:self.timeIndex],
            "fullStateHistory" : self.fullStateHistory[0:self.timeIndex],
            "controlActionHistory_L" : self.controlActionHistory_L[0:self.timeIndex],
            "AgentActionHistory" : self.AgentActionHistory[0:self.timeIndex],
            "constraintViolationHistory" : self.constraintViolationHistory[0:self.timeIndex],
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