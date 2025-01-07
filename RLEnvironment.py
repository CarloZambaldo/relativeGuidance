from imports import *

## REINFORCEMENT LEARNING ENVIRONMENT ##
class SimEnv(gym.Env):
	metadata = {"render_modes": ["none"]}

	def __init__(self):
		super(SimEnv,self).__init__()

		## OBSERVATION SPACE
		self.observation_space = spaces.Dict(
			{
				# "OBStateChaser_M": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
				# "OBStateTarget_M": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
				# "repulsiveAPFsurfaceValue": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
				"OBStateRelative_L": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
			}
		)

		## ACTION SPACE
		# 2 actions are present : 0 [skip Loop 1] or 1 [compute Loop 1]
		self.action_space = spaces.Discrete(2)
		self.trigger = True


	## STEP ##
	def step(self, AgentAction):

		# extract parameters for the current time step
		timeNow = self.timeHistory[self.timeIndex]

		# NAVIGATION # NOTE: this has already been computed for the current time step in previous cycle
		# indeed, the NAVIGATION is required for the agent to determine its action
		# self.OBStateTarget_M, _, self.OBStateRelative_L = OBNavigation(self.targetState_S, self.chaserState_S, self.param)

		# GUIDANCE ALGORITHM # 
		# compute the control action and output the optimal trajectory (if re-computed)
		executionTime_start = time.time()
		controlAction_L, self.OBoptimalTrajectory = \
								OBGuidance(timeNow, self.OBStateRelative_L, self.OBStateTarget_M,
									self.phaseID, self.param, AgentAction, self.OBoptimalTrajectory)		
		executionTime = time.time() - executionTime_start
		print(f"  > Guidance Step Execution Time: {executionTime*1e3:.2f} [ms]")
		print(f"  > Guidance Step Execution Time: {executionTime*1e3:.2f} [ms]")

		# CONTROL ACTION #
		# rotate the control action from the local frame to the synodic frame
		controlAction_S = OBControl(self.targetState_S,controlAction_L,self.param)

		print(f"control_L: {controlAction_L}")
		print(f"control_S: {controlAction_S}\n")

		# PHYSICAL ENVIRONMENT #
		# propagate the dynamics of the chaser for one time step (depends on Guidance Frequency)
		distAcceleration_S = ReferenceFrames.computeEnvironmentDisturbances(timeNow, self.param.chaser, self.param)
		odesol = solve_ivp(lambda t, state: dynamicsModel.CR3BP(t, state, self.param, controlAction_S, distAcceleration_S),
							  [timeNow, self.timeHistory[self.timeIndex + 1]], self.chaserState_S, method="DOP853", rtol=1e-11, atol=1e-11)
		self.fullStateHistory[self.timeIndex+1, 6:12] = odesol.y[:,-1] # extract following time step

		# REWARD COMPUTATION #
		self.reward, self.terminated = self.computeReward(AgentAction,controlAction_L,self.param)

		# PREPARE FOR NEXT TIME STEP #
		self.timeIndex += 1
		self.targetState_S = self.fullStateHistory[self.timeIndex,:6]
		self.chaserState_S = self.fullStateHistory[self.timeIndex,6:12]
		self.OBStateTarget_M, _, self.OBStateRelative_L = OBNavigation(self.targetState_S, self.chaserState_S, self.param)
		self.observation = self.computeRLobservation()

		info = {"param": self.param, "timeNow": timeNow}

		return self.observation, self.reward, self.terminated, self.truncated, info


	# The reset method should set the initial state of the environment (e.g., relative position and velocity) and return the initial observation.
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		# RL related parameters
		self.reward = 0
		self.terminated = False
		self.truncated = False

		# physical environment related parameters
		self.timeIndex = 0

		# guidance related parameters
		self.OBoptimalTrajectory = {}
		self.trigger = True

		# Set the mission phase
		if options is not None:
			self.phaseID = options["phaseID"]
		else:
			self.phaseID = 1
			#raise Warning("phaseID not defined. Setting it to 1")

		# Set the initial state of the pysical environment
		self.param, self.initialValue = config.env_config.get(seed,self.phaseID)

		# definition of the time vector with the GNC frequency
		self.timeHistory = np.arange(self.param.tspan[0], self.param.tspan[-1], 1/(self.param.freqGNC * self.param.tc))

		# INITIALIZATION OF THE MAIN VALUES FOR FULL SIMULATION HISTORY (definition of the solution vectors)
		self.controlActionHistory_L = np.zeros((len(self.timeHistory)+1, 3))
		self.fullStateHistory = np.zeros((len(self.timeHistory),12))

		# extraction of the initial conditions
		self.targetState_S = self.initialValue.fullInitialState[0:6]
		self.chaserState_S = self.initialValue.fullInitialState[6:12]

		# saving the initial states inside the fullStateHistory vector
		self.fullStateHistory[0,:] = self.initialValue.fullInitialState

		# integrate the dynamics of the target [for the whole simulation time]
		distAcceleration_S = dynamicsModel.computeEnvironmentDisturbances(0,self.param.target,self.param)
		odesol = solve_ivp(lambda t, state: dynamicsModel.CR3BP(t, state, self.param, distAcceleration_S),
								[self.timeHistory[0], self.timeHistory[-1]], self.targetState_S, t_eval=self.timeHistory,
								method="DOP853", rtol=1e-11, atol=1e-11)
		self.fullStateHistory[:, :6] = odesol.y.T # store the target dynamics

		## compute RL Agent Observation at time step 1
		self.OBStateTarget_M, _, self.OBStateRelative_L = OBNavigation(self.targetState_S, self.chaserState_S, self.param)
		
		info = {"param": self.param, "timeNow": self.param.tspan[0]}
		return self.computeRLobservation(), info









	## EXTRA METHODS ##
	def computeRLobservation(self):
		observation: dict = {
			"OBStateRelative_L" : self.OBStateRelative_L,
		} 
		return observation

	def computeReward(self,AgentAction,controlAction,param):
		if (AgentAction == 1):
			self.reward -= 1
			self.reward -= 1

		# if check.collision(self.chaserState_S-self.targetState_S):
		# 		terminated = True
		# 		reward = -1000

		if check.dockingSuccessful(self.chaserState_S-self.targetState_S, self.phaseID, self.param):
			terminated = True
			self.reward = 1000
		else:
			terminated = False

		# reduce the reward of an amount proportional to the Guidance control effort
		# (this strategy should reduce the computation of the that can lead to excessive control actions)
		self.reward -= np.linalg.norm(controlAction)*param.freqGNC

		return self.reward, terminated