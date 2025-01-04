from imports import *

## REINFORCEMENT LEARNING ENVIRONMENT ##
class SimEnv(gym.Env):
	# DO NOT FORGET TO ADD METADATA:
	metadata = {"render_modes": ["none"]}

	def __init__(self):
		super(SimEnv,self).__init__()

		## OBSERVATION SPACE
		# observations shall provide information about the location of the agent and target 
		# self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
		# Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
		self.observation_space = spaces.Dict(
			{
				# "OBStateChaser_M": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
				# "OBStateTarget_M": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
				# "repulsiveAPF": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
				"OBStateRelative_L": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
			}
		)

		## ACTION SPACE
		# 2 actions are present : OPTILOOP_COMP and OPTILOOP_SKIP 
		self.action_space = spaces.Discrete(2) # spaces.MultiDiscrete([2,2])

		self.trigger = True

	## STEP ##
	def step(self, AgentAction):
		
		# NAVIGATION #
		self.OBStateTarget_M, self.OBStateChaser_M, self.OBStateRelative_L = OBNavigation(self.stateChaser_M, self.stateTarget_M, self.stateRelative_L, self.param, self.trigger)
		
		# RL AGENT ACTION #
		if AgentAction == 1:
			self.trigger = True
		else:
			self.trigger = False

		# GUIDANCE ALGORITHM # relativeState_L, targetState_M, param
		self.controlAction_L = OBGuidance(self.envTime,self.OBStateRelative_L,self.OBStateTarget_M,self.initialValue.phaseID,trigger,param):
		
		# CONTROL ACTION #
		self.controlAction_S = OBControl(self.stateTarget_M,self.controlAction_L)

		# PHYSICAL ENVIRONMENT #
		

		# REWARD COMPUTATION #
		if collision_check(self.stateRelative_L):
			self.terminated = True
			self.reward = -10
			
		return self.observation, self.reward, self.terminated, self.truncated, self.info


	# The reset method should set the initial state of the environment (e.g., relative position and velocity) and return the initial observation.
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.terminated = False
		self.truncated = False
		self.envTime = 0 # seconds
		self.trigger = True

		# Set the mission phase
		if options is not None:
			self.phaseID = options["phaseID"]

		# Set the initial state of the pysical environment
		self.param, self.initialValue = config.env_config.get()

		
		return self
